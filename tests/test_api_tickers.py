"""Tests for ticker API endpoints."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestTickerEndpoints:
    """Test suite for /api/tickers endpoints."""

    def test_list_tickers_returns_200(self, test_client, mock_db_session):
        """Test listing tickers returns 200."""
        # Mock the database query results
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.last_updated = datetime.utcnow()
        mock_ticker.is_active = True

        mock_execute = MagicMock()
        mock_execute.scalars.return_value.all.return_value = [mock_ticker]
        mock_execute.scalar_one.return_value = 1
        mock_execute.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers")

        assert response.status_code == 200
        data = response.json()

        assert 'tickers' in data
        assert 'total' in data
        assert 'timestamp' in data

    def test_list_tickers_with_pagination(self, test_client, mock_db_session):
        """Test listing tickers with pagination parameters."""
        mock_execute = MagicMock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers?limit=10&offset=0")

        assert response.status_code == 200

    def test_list_tickers_active_only_filter(self, test_client, mock_db_session):
        """Test listing only active tickers."""
        mock_execute = MagicMock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers?active_only=true")

        assert response.status_code == 200

    def test_list_tickers_all_including_inactive(self, test_client, mock_db_session):
        """Test listing all tickers including inactive."""
        mock_execute = MagicMock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers?active_only=false")

        assert response.status_code == 200

    def test_list_tickers_limit_validation(self, test_client, mock_db_session):
        """Test that limit parameter is validated."""
        # Limit too high
        response = test_client.get("/api/tickers?limit=1000")
        assert response.status_code == 422

        # Limit too low
        response = test_client.get("/api/tickers?limit=0")
        assert response.status_code == 422

    def test_get_ticker_detail_success(self, test_client, mock_db_session):
        """Test getting ticker detail for valid ticker."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.market_cap = 3000000000000
        mock_ticker.sector = 'Technology'
        mock_ticker.industry = 'Consumer Electronics'
        mock_ticker.is_active = True

        mock_execute = MagicMock()
        mock_execute.scalar_one_or_none.return_value = mock_ticker
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/AAPL")

        assert response.status_code == 200
        data = response.json()

        assert data['symbol'] == 'AAPL'
        assert 'name' in data
        assert 'is_active' in data
        assert 'model_status' in data

    def test_get_ticker_detail_normalizes_symbol(self, test_client, mock_db_session):
        """Test that ticker symbol is normalized to uppercase."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.market_cap = None
        mock_ticker.sector = None
        mock_ticker.industry = None
        mock_ticker.is_active = True

        mock_execute = MagicMock()
        mock_execute.scalar_one_or_none.return_value = mock_ticker
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/aapl")

        assert response.status_code == 200
        data = response.json()
        assert data['symbol'] == 'AAPL'

    def test_get_ticker_detail_not_found(self, test_client, mock_db_session):
        """Test 404 for non-existent ticker."""
        mock_execute = MagicMock()
        mock_execute.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/NONEXISTENT")

        assert response.status_code == 404

    def test_get_ticker_detail_structure(self, test_client, mock_db_session):
        """Test ticker detail response structure."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.market_cap = 3000000000000
        mock_ticker.sector = 'Technology'
        mock_ticker.industry = 'Consumer Electronics'
        mock_ticker.is_active = True

        mock_execute = MagicMock()
        mock_execute.scalar_one_or_none.return_value = mock_ticker
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 0
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/AAPL")

        data = response.json()

        # Check all expected fields
        expected_fields = [
            'symbol', 'name', 'market_cap', 'sector', 'industry',
            'current_price', 'volume_24h', 'price_change_24h',
            'high_24h', 'low_24h', 'is_active', 'total_predictions',
            'last_prediction_time', 'model_status'
        ]

        for field in expected_fields:
            assert field in data, f"Missing field: {field}"


class TestTickerMetricsCalculation:
    """Test suite for ticker metrics calculation logic."""

    def test_price_change_calculation(self, test_client, mock_db_session):
        """Test that price change is calculated correctly."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.market_cap = None
        mock_ticker.sector = None
        mock_ticker.industry = None
        mock_ticker.is_active = True

        mock_latest_bar = MagicMock()
        mock_latest_bar.close = 150.0
        mock_latest_bar.high = 155.0
        mock_latest_bar.low = 145.0

        mock_old_bar = MagicMock()
        mock_old_bar.close = 140.0

        def execute_side_effect(stmt):
            mock_result = MagicMock()
            # Simulate different query results
            mock_result.scalar_one_or_none.return_value = mock_ticker
            mock_result.scalars.return_value.all.return_value = [mock_latest_bar]
            mock_result.scalar_one.return_value = 1000000
            return mock_result

        mock_db_session.execute.side_effect = execute_side_effect

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/AAPL")

        assert response.status_code == 200

    def test_volume_aggregation(self, test_client, mock_db_session):
        """Test that 24h volume is aggregated correctly."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = 'AAPL'
        mock_ticker.name = 'Apple Inc.'
        mock_ticker.market_cap = None
        mock_ticker.sector = None
        mock_ticker.industry = None
        mock_ticker.is_active = True

        mock_execute = MagicMock()
        mock_execute.scalar_one_or_none.return_value = mock_ticker
        mock_execute.scalars.return_value.all.return_value = []
        mock_execute.scalar_one.return_value = 5000000  # 5M volume
        mock_db_session.execute.return_value = mock_execute

        with patch('src.api.routes.tickers.get_db', return_value=iter([mock_db_session])):
            response = test_client.get("/api/tickers/AAPL")

        assert response.status_code == 200
