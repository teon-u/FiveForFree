"""Tests for prediction API endpoints."""
import pytest
from datetime import datetime


class TestPredictionEndpoints:
    """Test suite for /api/predictions endpoints."""

    def test_get_single_prediction_success(self, test_client):
        """Test getting prediction for a valid ticker."""
        response = test_client.get("/api/predictions/AAPL")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert data['ticker'] == 'AAPL'
        assert 'timestamp' in data
        assert 'current_price' in data
        assert 'up_probability' in data
        assert 'down_probability' in data
        assert 'best_up_model' in data
        assert 'best_down_model' in data
        assert 'trading_signal' in data
        assert 'confidence_level' in data

    def test_get_prediction_normalizes_ticker_to_uppercase(self, test_client):
        """Test that ticker is normalized to uppercase."""
        response = test_client.get("/api/predictions/aapl")

        assert response.status_code == 200
        data = response.json()
        assert data['ticker'] == 'AAPL'

    def test_get_prediction_invalid_ticker_returns_404(self, test_client):
        """Test that invalid ticker returns 404."""
        response = test_client.get("/api/predictions/INVALID")

        assert response.status_code == 404

    def test_get_prediction_with_all_models_flag(self, test_client):
        """Test prediction with include_all_models parameter."""
        response = test_client.get("/api/predictions/AAPL?include_all_models=true")

        assert response.status_code == 200

    def test_get_prediction_with_context_flag(self, test_client):
        """Test prediction with include_context parameter."""
        response = test_client.get("/api/predictions/AAPL?include_context=true")

        assert response.status_code == 200

    def test_get_prediction_trading_signal_values(self, test_client):
        """Test that trading signal is one of BUY, SELL, HOLD."""
        response = test_client.get("/api/predictions/AAPL")
        data = response.json()

        assert data['trading_signal'] in ['BUY', 'SELL', 'HOLD']

    def test_get_prediction_confidence_level_values(self, test_client):
        """Test that confidence level is valid."""
        response = test_client.get("/api/predictions/AAPL")
        data = response.json()

        assert data['confidence_level'] in ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW']

    def test_get_prediction_probabilities_in_range(self, test_client):
        """Test that probabilities are between 0 and 1."""
        response = test_client.get("/api/predictions/AAPL")
        data = response.json()

        assert 0 <= data['up_probability'] <= 1
        assert 0 <= data['down_probability'] <= 1

    def test_batch_predictions_success(self, test_client):
        """Test batch predictions with multiple tickers."""
        response = test_client.post(
            "/api/predictions",
            json={"tickers": ["AAPL", "GOOGL", "MSFT"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert 'predictions' in data
        assert 'total_requested' in data
        assert 'total_successful' in data
        assert 'failed_tickers' in data
        assert 'timestamp' in data

        assert data['total_requested'] == 3
        assert data['total_successful'] >= 1

    def test_batch_predictions_normalizes_tickers(self, test_client):
        """Test that batch predictions normalize tickers to uppercase."""
        response = test_client.post(
            "/api/predictions",
            json={"tickers": ["aapl", "googl"]}
        )

        assert response.status_code == 200
        data = response.json()

        # Keys should be uppercase
        for ticker in data['predictions'].keys():
            assert ticker == ticker.upper()

    def test_batch_predictions_handles_invalid_tickers(self, test_client):
        """Test that batch predictions handles mix of valid and invalid tickers."""
        response = test_client.post(
            "/api/predictions",
            json={"tickers": ["AAPL", "INVALID1", "GOOGL", "INVALID2"]}
        )

        assert response.status_code == 200
        data = response.json()

        assert data['total_requested'] == 4
        assert len(data['failed_tickers']) >= 2

    def test_batch_predictions_empty_list_returns_error(self, test_client):
        """Test that empty ticker list returns validation error."""
        response = test_client.post(
            "/api/predictions",
            json={"tickers": []}
        )

        assert response.status_code == 422  # Validation error

    def test_batch_predictions_with_include_all_models(self, test_client):
        """Test batch predictions with include_all_models flag."""
        response = test_client.post(
            "/api/predictions",
            json={"tickers": ["AAPL"], "include_all_models": True}
        )

        assert response.status_code == 200

    def test_top_opportunities_up_direction(self, test_client):
        """Test getting top opportunities for up direction."""
        response = test_client.get(
            "/api/predictions/top/opportunities",
            params={"tickers": ["AAPL", "GOOGL", "MSFT"], "direction": "up"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data['direction'] == 'up'
        assert 'opportunities' in data
        assert 'total_signals' in data
        assert 'timestamp' in data

    def test_top_opportunities_down_direction(self, test_client):
        """Test getting top opportunities for down direction."""
        response = test_client.get(
            "/api/predictions/top/opportunities",
            params={"tickers": ["AAPL", "GOOGL"], "direction": "down"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data['direction'] == 'down'

    def test_top_opportunities_with_min_probability(self, test_client):
        """Test top opportunities with minimum probability filter."""
        response = test_client.get(
            "/api/predictions/top/opportunities",
            params={"tickers": ["AAPL"], "direction": "up", "min_probability": 0.8}
        )

        assert response.status_code == 200

    def test_top_opportunities_with_top_n_limit(self, test_client):
        """Test top opportunities with top_n limit."""
        response = test_client.get(
            "/api/predictions/top/opportunities",
            params={"tickers": ["AAPL", "GOOGL", "MSFT"], "direction": "up", "top_n": 2}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data['opportunities']) <= 2

    def test_top_opportunities_invalid_direction_returns_error(self, test_client):
        """Test that invalid direction returns validation error."""
        response = test_client.get(
            "/api/predictions/top/opportunities",
            params={"tickers": ["AAPL"], "direction": "invalid"}
        )

        assert response.status_code == 422

    def test_categorized_predictions_success(self, test_client):
        """Test getting categorized predictions."""
        response = test_client.get("/api/predictions")

        assert response.status_code == 200
        data = response.json()

        assert 'volume_top_100' in data
        assert 'gainers_top_100' in data
        assert 'timestamp' in data

    def test_categorized_predictions_with_threshold(self, test_client):
        """Test categorized predictions with probability threshold."""
        response = test_client.get("/api/predictions?threshold=0.8")

        assert response.status_code == 200

    def test_categorized_predictions_structure(self, test_client):
        """Test that categorized predictions have correct item structure."""
        response = test_client.get("/api/predictions")
        data = response.json()

        if data['volume_top_100']:
            item = data['volume_top_100'][0]
            assert 'ticker' in item
            assert 'probability' in item
            assert 'direction' in item
            assert 'best_model' in item
            assert 'hit_rate' in item
            assert 'current_price' in item
