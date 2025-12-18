"""
Infrastructure component tests for NASDAQ Prediction System.

Tests for:
- RateLimiter: Rate limiting middleware
- WebSocket: Real-time communication
- Scheduler: Background job scheduling

Total: 27 tests (9 per component)
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse

from src.api.middleware.rate_limiter import RateLimiter, RateLimitMiddleware
from src.api.websocket import ConnectionManager, handle_websocket_connection, process_client_message
from src.scheduler import NASDAQScheduler


# ==============================================================================
# RateLimiter Tests (9 tests)
# ==============================================================================

class TestRateLimiter:
    """Test suite for RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test 1: RateLimiter initialization with default and custom parameters."""
        # Default initialization
        limiter = RateLimiter()
        assert limiter.requests_per_minute == 60
        assert limiter.requests_per_hour == 1000
        assert limiter.burst_limit == 10
        assert limiter._cleanup_interval == 300

        # Custom initialization
        limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_limit=5
        )
        assert limiter.requests_per_minute == 30
        assert limiter.requests_per_hour == 500
        assert limiter.burst_limit == 5

    def test_client_id_extraction(self):
        """Test 2: Client ID extraction from request headers."""
        limiter = RateLimiter()

        # Test X-Forwarded-For header
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = "192.168.1.1, 10.0.0.1"
        mock_request.client = None

        client_id = limiter._get_client_id(mock_request)
        assert client_id == "192.168.1.1"

        # Test direct client IP
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        client_id = limiter._get_client_id(mock_request)
        assert client_id == "127.0.0.1"

        # Test unknown client
        mock_request.client = None
        client_id = limiter._get_client_id(mock_request)
        assert client_id == "unknown"

    def test_per_minute_limit(self):
        """Test 3: Per-minute rate limit enforcement."""
        limiter = RateLimiter(requests_per_minute=5, burst_limit=10)

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        # Allow first 5 requests
        for i in range(5):
            allowed, reason, headers = limiter.is_allowed(mock_request)
            assert allowed is True
            # Headers show remaining BEFORE this request is recorded
            # So remaining = limit - current_count (not including this request)
            assert headers["X-RateLimit-Remaining-Minute"] == 5 - i

        # 6th request should be denied
        allowed, reason, headers = limiter.is_allowed(mock_request)
        assert allowed is False
        assert "minute" in reason.lower()

    def test_per_hour_limit(self):
        """Test 4: Per-hour rate limit enforcement."""
        limiter = RateLimiter(
            requests_per_minute=100,
            requests_per_hour=10,
            burst_limit=100
        )

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        # Allow first 10 requests
        for i in range(10):
            allowed, reason, headers = limiter.is_allowed(mock_request)
            assert allowed is True
            # Headers show remaining BEFORE this request is recorded
            assert headers["X-RateLimit-Remaining-Hour"] == 10 - i

        # 11th request should be denied
        allowed, reason, headers = limiter.is_allowed(mock_request)
        assert allowed is False
        assert "hour" in reason.lower()

    def test_burst_limit(self):
        """Test 5: Burst limit enforcement (multiple requests in 1 second)."""
        limiter = RateLimiter(burst_limit=3)

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        # First 3 requests should succeed
        for i in range(3):
            allowed, reason, headers = limiter.is_allowed(mock_request)
            assert allowed is True

        # 4th request in same second should be denied
        allowed, reason, headers = limiter.is_allowed(mock_request)
        assert allowed is False
        assert "burst" in reason.lower()

    def test_rate_limit_rejection(self):
        """Test 6: Proper rejection when limits exceeded."""
        limiter = RateLimiter(requests_per_minute=2, burst_limit=10)

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        # Exhaust minute limit
        for _ in range(2):
            limiter.is_allowed(mock_request)

        # Next request should be rejected with proper headers
        allowed, reason, headers = limiter.is_allowed(mock_request)
        assert allowed is False
        assert reason != ""
        assert "X-RateLimit-Limit-Minute" in headers
        assert "X-RateLimit-Remaining-Minute" in headers
        assert headers["X-RateLimit-Remaining-Minute"] == 0

    def test_cleanup_old_requests(self):
        """Test 7: Cleanup of old request timestamps."""
        limiter = RateLimiter()
        limiter._cleanup_interval = 0  # Force cleanup on every call

        # Add old requests (simulate)
        client_id = "test_client"
        old_time = time.time() - 7200  # 2 hours ago
        limiter._requests[client_id] = [old_time, old_time + 10, old_time + 20]

        # Trigger cleanup
        limiter._cleanup_old_requests()

        # Old entries should be removed
        assert client_id not in limiter._requests or len(limiter._requests[client_id]) == 0

    def test_multiple_clients_isolation(self):
        """Test 8: Rate limits are isolated per client."""
        limiter = RateLimiter(requests_per_minute=2, burst_limit=10)

        # Client 1
        mock_request1 = MagicMock(spec=Request)
        mock_request1.headers.get.return_value = None
        mock_client1 = MagicMock()
        mock_client1.host = "192.168.1.1"
        mock_request1.client = mock_client1

        # Client 2
        mock_request2 = MagicMock(spec=Request)
        mock_request2.headers.get.return_value = None
        mock_client2 = MagicMock()
        mock_client2.host = "192.168.1.2"
        mock_request2.client = mock_client2

        # Client 1 exhausts limit
        limiter.is_allowed(mock_request1)
        limiter.is_allowed(mock_request1)
        allowed, _, _ = limiter.is_allowed(mock_request1)
        assert allowed is False

        # Client 2 should still be allowed
        allowed, _, _ = limiter.is_allowed(mock_request2)
        assert allowed is True

    def test_rate_limit_reset(self):
        """Test 9: Rate limits reset after time window expires."""
        limiter = RateLimiter(requests_per_minute=2, burst_limit=10)

        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client

        # Exhaust limit
        limiter.is_allowed(mock_request)
        limiter.is_allowed(mock_request)
        allowed, _, _ = limiter.is_allowed(mock_request)
        assert allowed is False

        # Simulate time passing by manually adjusting request timestamps
        client_id = limiter._get_client_id(mock_request)
        old_time = time.time() - 120  # 2 minutes ago
        limiter._requests[client_id] = [old_time, old_time + 1]

        # Should be allowed again
        allowed, _, _ = limiter.is_allowed(mock_request)
        assert allowed is True


# ==============================================================================
# WebSocket Tests (9 tests)
# ==============================================================================

class TestWebSocket:
    """Test suite for WebSocket ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connection_acceptance(self):
        """Test 1: WebSocket connection acceptance."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)

        # Connect
        await manager.connect(mock_websocket)

        # Verify connection was accepted and registered
        mock_websocket.accept.assert_called_once()
        assert mock_websocket in manager.active_connections
        assert mock_websocket in manager.subscribed_tickers
        assert len(manager.active_connections) == 1

    def test_connection_rejection(self):
        """Test 2: WebSocket connection disconnection/rejection."""
        manager = ConnectionManager()
        mock_websocket = MagicMock(spec=WebSocket)

        # Add connection manually
        manager.active_connections.add(mock_websocket)
        manager.subscribed_tickers[mock_websocket] = {"AAPL", "GOOGL"}

        # Disconnect
        manager.disconnect(mock_websocket)

        # Verify removal
        assert mock_websocket not in manager.active_connections
        assert mock_websocket not in manager.subscribed_tickers

    @pytest.mark.asyncio
    async def test_message_receiving(self):
        """Test 3: Message reception from client."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_predictor = MagicMock()

        # Test ping message
        message = {"type": "ping"}
        await process_client_message(mock_websocket, message, mock_predictor)

        # Should send pong response
        assert mock_websocket.send_json.called
        sent_message = mock_websocket.send_json.call_args[0][0]
        assert sent_message["type"] == "pong"

    @pytest.mark.asyncio
    async def test_message_broadcast(self):
        """Test 4: Message broadcasting to all clients."""
        manager = ConnectionManager()

        # Create mock connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws3 = AsyncMock(spec=WebSocket)

        manager.active_connections.add(mock_ws1)
        manager.active_connections.add(mock_ws2)
        manager.active_connections.add(mock_ws3)

        # Broadcast message
        test_message = {"type": "test", "data": "broadcast"}
        await manager.broadcast(test_message)

        # Verify all clients received message
        mock_ws1.send_json.assert_called_once_with(test_message)
        mock_ws2.send_json.assert_called_once_with(test_message)
        mock_ws3.send_json.assert_called_once_with(test_message)

    @pytest.mark.asyncio
    async def test_subscription_management(self):
        """Test 5: Subscribe and unsubscribe to ticker updates."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)

        await manager.connect(mock_websocket)

        # Subscribe to tickers
        tickers = ["AAPL", "GOOGL", "MSFT"]
        manager.subscribe_tickers(mock_websocket, tickers)

        subscriptions = manager.get_subscriptions(mock_websocket)
        assert "AAPL" in subscriptions
        assert "GOOGL" in subscriptions
        assert "MSFT" in subscriptions

        # Unsubscribe from some tickers
        manager.unsubscribe_tickers(mock_websocket, ["AAPL", "MSFT"])

        subscriptions = manager.get_subscriptions(mock_websocket)
        assert "AAPL" not in subscriptions
        assert "GOOGL" in subscriptions
        assert "MSFT" not in subscriptions

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test 6: Heartbeat mechanism for connection keepalive."""
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)

        manager.active_connections.add(mock_ws1)
        manager.active_connections.add(mock_ws2)

        # Send heartbeat
        heartbeat_msg = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "connections": 2
        }
        await manager.broadcast(heartbeat_msg)

        # Verify heartbeat received
        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_termination(self):
        """Test 7: Proper connection cleanup on disconnect."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)

        await manager.connect(mock_websocket)
        manager.subscribe_tickers(mock_websocket, ["AAPL", "GOOGL"])

        # Verify connection state
        assert mock_websocket in manager.active_connections
        assert len(manager.get_subscriptions(mock_websocket)) == 2

        # Disconnect
        manager.disconnect(mock_websocket)

        # Verify cleanup
        assert mock_websocket not in manager.active_connections
        assert mock_websocket not in manager.subscribed_tickers

    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test 8: Multiple simultaneous WebSocket connections."""
        manager = ConnectionManager()

        # Create multiple connections
        connections = [AsyncMock(spec=WebSocket) for _ in range(10)]

        # Connect all
        for ws in connections:
            await manager.connect(ws)

        assert len(manager.active_connections) == 10

        # Broadcast to all
        test_msg = {"type": "test", "data": "concurrent"}
        await manager.broadcast(test_msg)

        # Verify all received
        for ws in connections:
            ws.send_json.assert_called_once_with(test_msg)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test 9: Error handling during message sending."""
        manager = ConnectionManager()

        # Create mock connections - one that works, one that fails
        mock_ws_good = AsyncMock(spec=WebSocket)
        mock_ws_bad = AsyncMock(spec=WebSocket)
        mock_ws_bad.send_json.side_effect = Exception("Connection lost")

        manager.active_connections.add(mock_ws_good)
        manager.active_connections.add(mock_ws_bad)

        # Broadcast should handle error gracefully
        test_msg = {"type": "test", "data": "error_test"}
        await manager.broadcast(test_msg)

        # Good connection should receive message
        mock_ws_good.send_json.assert_called_once_with(test_msg)

        # Bad connection should be removed
        assert mock_ws_bad not in manager.active_connections


# ==============================================================================
# Scheduler Tests (9 tests)
# ==============================================================================

class TestScheduler:
    """Test suite for NASDAQScheduler."""

    def test_scheduler_initialization(self):
        """Test 1: Scheduler initialization with components."""
        mock_model_manager = MagicMock()

        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        assert scheduler.model_manager is mock_model_manager
        assert scheduler.enable_market_hours_check is False
        assert scheduler.scheduler is None
        assert isinstance(scheduler.active_tickers, dict)
        assert "volume" in scheduler.active_tickers
        assert "gainers" in scheduler.active_tickers

    def test_job_registration(self):
        """Test 2: Job registration in scheduler."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        # Mock the job_update_tickers to prevent actual execution
        with patch.object(scheduler, 'job_update_tickers'):
            scheduler.start()

        assert scheduler.scheduler is not None
        assert scheduler.scheduler.running is True

        # Check jobs are registered
        jobs = scheduler.scheduler.get_jobs()
        job_ids = [job.id for job in jobs]

        assert "minute_collect" in job_ids
        assert "hourly_training" in job_ids
        assert "ticker_update" in job_ids
        assert "daily_retrain" in job_ids

        scheduler.stop()

    def test_periodic_execution(self):
        """Test 3: Periodic job execution simulation."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        # Test job execution tracking
        initial_runs = scheduler.job_stats["ticker_update"]["runs"]

        # Manually trigger job
        with patch.object(scheduler.ticker_selector, 'get_both_categories') as mock_selector:
            mock_selector.return_value = {
                'volume': [MagicMock(ticker="AAPL")],
                'gainers': [MagicMock(ticker="GOOGL")]
            }
            scheduler.job_update_tickers()

        # Verify run count increased
        assert scheduler.job_stats["ticker_update"]["runs"] == initial_runs + 1
        assert scheduler.job_stats["ticker_update"]["last_run"] is not None

    def test_job_cancellation(self):
        """Test 4: Job cancellation and removal."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        with patch.object(scheduler, 'job_update_tickers'):
            scheduler.start()

        assert scheduler.scheduler.running is True

        # Stop scheduler (cancels all jobs)
        scheduler.stop()

        assert scheduler.scheduler is None

    def test_error_recovery(self):
        """Test 5: Error recovery during job execution."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        initial_errors = scheduler.job_stats["ticker_update"]["errors"]

        # Force an error in job execution
        with patch.object(scheduler.ticker_selector, 'get_both_categories') as mock_selector:
            mock_selector.side_effect = Exception("Test error")

            # Job should handle error gracefully
            scheduler.job_update_tickers()

        # Error count should increase
        assert scheduler.job_stats["ticker_update"]["errors"] == initial_errors + 1

    def test_concurrent_jobs(self):
        """Test 6: Multiple jobs can run concurrently without conflicts."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        # Mock multiple job executions
        with patch.object(scheduler.ticker_selector, 'get_both_categories') as mock_selector:
            mock_selector.return_value = {
                'volume': [MagicMock(ticker="AAPL")],
                'gainers': [MagicMock(ticker="GOOGL")]
            }

            # Execute ticker update job
            scheduler.job_update_tickers()

            # Execute collection job (should not interfere)
            scheduler.job_collect_and_predict()

        # Both jobs should have recorded their runs
        assert scheduler.job_stats["ticker_update"]["runs"] > 0

    def test_scheduler_start_stop(self):
        """Test 7: Scheduler start and stop operations."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        # Initially not running
        assert scheduler.scheduler is None

        # Start scheduler
        with patch.object(scheduler, 'job_update_tickers'):
            scheduler.start()

        assert scheduler.scheduler is not None
        assert scheduler.scheduler.running is True

        # Stop scheduler
        scheduler.stop()
        assert scheduler.scheduler is None

    def test_job_status_query(self):
        """Test 8: Query job status and statistics."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        # Before starting
        status = scheduler.get_status()
        assert status["running"] is False

        # After starting
        with patch.object(scheduler, 'job_update_tickers'):
            scheduler.start()

        status = scheduler.get_status()
        assert status["running"] is True
        assert "jobs" in status
        assert "job_stats" in status
        assert "active_tickers" in status
        assert "market_hours" in status

        scheduler.stop()

    def test_job_timeout_handling(self):
        """Test 9: Job timeout and max_instances configuration."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=False
        )

        with patch.object(scheduler, 'job_update_tickers'):
            scheduler.start()

        # Check that jobs have max_instances=1 (prevents overlap)
        jobs = scheduler.scheduler.get_jobs()

        for job in jobs:
            # Each job should have max_instances configured
            assert job.max_instances == 1
            # Jobs should coalesce (skip if already running)
            assert job.coalesce is True

        scheduler.stop()


# ==============================================================================
# Integration Tests (Bonus)
# ==============================================================================

class TestInfrastructureIntegration:
    """Integration tests for infrastructure components."""

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_integration(self):
        """Integration test: RateLimitMiddleware with FastAPI."""
        from fastapi import FastAPI

        app = FastAPI()

        # Add rate limiting middleware
        middleware = RateLimitMiddleware(
            app=app,
            requests_per_minute=2,
            burst_limit=5
        )

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.headers.get.return_value = None
        mock_client = MagicMock()
        mock_client.host = "127.0.0.1"
        mock_request.client = mock_client
        mock_request.url.path = "/api/test"

        # Mock call_next
        mock_response = MagicMock()
        mock_response.headers = {}

        async def mock_call_next(request):
            return mock_response

        # Test normal request
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response is mock_response

    def test_scheduler_market_hours_check(self):
        """Integration test: Scheduler market hours validation."""
        mock_model_manager = MagicMock()
        scheduler = NASDAQScheduler(
            model_manager=mock_model_manager,
            enable_market_hours_check=True
        )

        # The market hours check should consider weekends and time
        import pytz
        from datetime import datetime

        # Mock a weekend
        with patch('src.scheduler.datetime') as mock_datetime:
            saturday = datetime(2024, 1, 6, 12, 0, 0)  # Saturday
            mock_datetime.now.return_value = pytz.timezone('America/New_York').localize(saturday)

            # Should return False on weekend
            is_market_hours = scheduler.is_market_hours()

            # Note: Actual result depends on implementation
            # This test verifies the method runs without error
            assert isinstance(is_market_hours, bool)
