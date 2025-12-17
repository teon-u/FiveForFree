"""Tests for health check API endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestHealthEndpoints:
    """Test suite for /api/health endpoints."""

    def test_health_check_returns_200(self, test_client, mock_market_status):
        """Test main health check endpoint returns 200 with correct structure."""
        with patch('src.api.routes.health.get_market_status', return_value=mock_market_status):
            response = test_client.get("/api/health")

            assert response.status_code == 200
            data = response.json()

            # Check required fields
            assert data['status'] == 'healthy'
            assert 'timestamp' in data
            assert 'version' in data
            assert 'database' in data
            assert 'models' in data
            assert 'market' in data

            # Check models structure
            assert 'total_tickers' in data['models']
            assert 'total_models' in data['models']
            assert 'trained_models' in data['models']

    def test_health_check_with_trailing_slash(self, test_client, mock_market_status):
        """Test health check with trailing slash."""
        with patch('src.api.routes.health.get_market_status', return_value=mock_market_status):
            response = test_client.get("/api/health/")
            assert response.status_code == 200

    def test_readiness_check_returns_ready_status(self, test_client):
        """Test readiness check returns ready status when models are trained."""
        response = test_client.get("/api/health/ready")

        assert response.status_code == 200
        data = response.json()

        assert 'ready' in data
        assert 'timestamp' in data
        assert 'details' in data
        assert 'trained_models' in data['details']
        assert 'total_models' in data['details']

    def test_readiness_check_ready_when_trained_models_exist(self, test_client):
        """Test readiness returns True when trained models > 0."""
        response = test_client.get("/api/health/ready")

        data = response.json()
        # Our mock returns trained_models > 0
        assert data['ready'] is True

    def test_liveness_check_returns_alive(self, test_client):
        """Test liveness check returns alive status."""
        response = test_client.get("/api/health/live")

        assert response.status_code == 200
        data = response.json()

        assert data['status'] == 'alive'
        assert 'timestamp' in data

    def test_health_check_market_status_structure(self, test_client, mock_market_status):
        """Test that market status has correct structure."""
        with patch('src.api.routes.health.get_market_status', return_value=mock_market_status):
            response = test_client.get("/api/health")
            data = response.json()

            market = data['market']
            assert 'is_open' in market
            assert 'is_trading_day' in market
            assert 'current_time_et' in market
            assert 'market_open' in market
            assert 'market_close' in market

    def test_health_check_version_format(self, test_client, mock_market_status):
        """Test that version follows semver format."""
        with patch('src.api.routes.health.get_market_status', return_value=mock_market_status):
            response = test_client.get("/api/health")
            data = response.json()

            version = data['version']
            parts = version.split('.')
            assert len(parts) == 3
            assert all(part.isdigit() for part in parts)

    def test_health_check_timestamp_is_iso_format(self, test_client, mock_market_status):
        """Test that timestamp is in ISO format."""
        with patch('src.api.routes.health.get_market_status', return_value=mock_market_status):
            response = test_client.get("/api/health")
            data = response.json()

            # Should not raise an exception if valid ISO format
            datetime.fromisoformat(data['timestamp'])
