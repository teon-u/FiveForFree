"""Pytest configuration and fixtures for API testing."""
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient


# Mock classes for testing
class MockModel:
    """Mock model for testing."""

    def __init__(self, ticker="TEST", target="up", model_type="xgboost"):
        self.ticker = ticker
        self.target = target
        self.model_type = model_type
        self.is_trained = True
        self.prediction_history = []

    def predict_proba(self, X):
        return np.random.rand(len(X))

    def get_recent_accuracy(self, hours=50):
        return 0.75

    def get_prediction_stats(self, hours=50):
        return {
            'accuracy': 0.75,
            'precision': 0.72,
            'recall': 0.78,
            'f1_score': 0.75,
            'true_positives': 15,
            'false_positives': 5,
            'true_negatives': 10,
            'false_negatives': 5,
            'total_predictions': 35,
            'avg_probability': 0.68
        }

    def get_roc_curve_data(self, hours=50):
        return {
            'fpr': [0.0, 0.1, 0.2, 0.5, 1.0],
            'tpr': [0.0, 0.5, 0.7, 0.9, 1.0],
            'thresholds': [1.0, 0.8, 0.6, 0.4, 0.0],
            'auc': 0.85
        }

    def get_precision_recall_curve(self, hours=50):
        return {
            'precision': [1.0, 0.9, 0.8, 0.7, 0.6],
            'recall': [0.0, 0.3, 0.5, 0.7, 1.0],
            'thresholds': [0.9, 0.7, 0.5, 0.3, 0.1],
            'ap': 0.82
        }

    def get_performance_over_time(self, hours=50, window_hours=5):
        return [
            {'timestamp': (datetime.utcnow() - timedelta(hours=i*5)).isoformat(), 'accuracy': 0.7 + np.random.rand() * 0.1}
            for i in range(10)
        ]

    def get_calibration_curve(self, hours=50, n_bins=10):
        return {
            'mean_predicted_proba': [0.1 * i for i in range(1, 11)],
            'fraction_positive': [0.08 + 0.08 * i for i in range(10)],
            'counts': [10] * 10
        }

    def get_backtest_results(self, hours=50):
        return {
            'metrics': {
                'sharpe_ratio': 1.8,
                'sortino_ratio': 2.1,
                'max_drawdown': -0.05,
                'calmar_ratio': 1.2,
                'total_trades': 20,
                'win_rate': 0.65,
                'total_return': 0.15,
                'avg_return': 0.0075,
                'avg_win': 0.02,
                'avg_loss': -0.01,
                'best_trade': 0.05,
                'worst_trade': -0.03,
                'profit_factor': 2.5
            },
            'equity_curve': [
                {'timestamp': (datetime.utcnow() - timedelta(hours=i)).isoformat(), 'equity': 1.0 + 0.003 * i}
                for i in range(50)
            ],
            'trade_distribution': [
                {'bucket': '-3% to -2%', 'count': 2},
                {'bucket': '-2% to -1%', 'count': 3},
                {'bucket': '-1% to 0%', 'count': 5},
                {'bucket': '0% to 1%', 'count': 4},
                {'bucket': '1% to 2%', 'count': 4},
                {'bucket': '2% to 3%', 'count': 2}
            ],
            'recent_trades': [
                {'timestamp': datetime.utcnow().isoformat(), 'return': 0.02, 'direction': 'up'},
                {'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(), 'return': -0.01, 'direction': 'up'}
            ]
        }


class MockEnsembleModel(MockModel):
    """Mock ensemble model for testing."""

    def get_base_model_weights(self):
        return {
            'xgboost': 0.3,
            'lightgbm': 0.25,
            'lstm': 0.25,
            'transformer': 0.2
        }

    def get_ensemble_stats(self):
        return {
            'meta_learner_type': 'LogisticRegression',
            'trained_base_models': 4,
            'total_base_models': 4
        }


class MockModelManager:
    """Mock model manager for testing."""

    def __init__(self):
        self._models = {}
        self._tickers = ['AAPL', 'GOOGL', 'MSFT']

    def get_summary(self):
        return {
            'total_tickers': len(self._tickers),
            'tickers': self._tickers,
            'total_models': len(self._tickers) * 10,  # 5 model types * 2 directions
            'trained_models': len(self._tickers) * 8,
            'untrained_models': len(self._tickers) * 2,
            'models_path': '/tmp/models'
        }

    def get_tickers(self):
        return self._tickers

    def get_best_model(self, ticker, target):
        if ticker.upper() not in [t.upper() for t in self._tickers]:
            raise ValueError(f"No models for {ticker}")
        return 'xgboost', MockModel(ticker, target, 'xgboost')

    def get_or_create_model(self, ticker, model_type, target):
        if model_type == 'ensemble':
            return 'ensemble', MockEnsembleModel(ticker, target, 'ensemble')
        return model_type, MockModel(ticker, target, model_type)

    def get_model_performances(self, ticker):
        return {
            'up': {
                'xgboost': {'hit_rate_50h': 75.0, 'total_predictions': 100, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'lightgbm': {'hit_rate_50h': 72.0, 'total_predictions': 95, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'lstm': {'hit_rate_50h': 68.0, 'total_predictions': 90, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'transformer': {'hit_rate_50h': 70.0, 'total_predictions': 85, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'ensemble': {'hit_rate_50h': 78.0, 'total_predictions': 100, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()}
            },
            'down': {
                'xgboost': {'hit_rate_50h': 73.0, 'total_predictions': 100, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'lightgbm': {'hit_rate_50h': 70.0, 'total_predictions': 95, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'lstm': {'hit_rate_50h': 65.0, 'total_predictions': 90, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'transformer': {'hit_rate_50h': 67.0, 'total_predictions': 85, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()},
                'ensemble': {'hit_rate_50h': 76.0, 'total_predictions': 100, 'is_trained': True, 'last_trained': datetime.utcnow().isoformat()}
            }
        }

    def validate_models(self, ticker):
        return {
            'xgboost_up': True,
            'xgboost_down': True,
            'lightgbm_up': True,
            'lightgbm_down': True,
            'lstm_up': True,
            'lstm_down': True,
            'transformer_up': True,
            'transformer_down': True,
            'ensemble_up': True,
            'ensemble_down': True
        }


class MockPredictionResult:
    """Mock prediction result."""

    def __init__(self, ticker):
        self.ticker = ticker
        self.timestamp = datetime.utcnow()
        self.current_price = 150.0 + np.random.rand() * 50
        self.up_probability = 0.7 + np.random.rand() * 0.2
        self.down_probability = 0.3 + np.random.rand() * 0.2
        self.best_up_model = 'xgboost'
        self.best_down_model = 'lightgbm'
        self.up_model_accuracy = 0.75
        self.down_model_accuracy = 0.72
        self.all_model_predictions = None
        self.market_context = None

    def get_trading_signal(self):
        if self.up_probability >= 0.7:
            return 'BUY'
        elif self.down_probability >= 0.7:
            return 'SELL'
        return 'HOLD'

    def get_confidence_level(self):
        max_prob = max(self.up_probability, self.down_probability)
        if max_prob >= 0.85:
            return 'VERY_HIGH'
        elif max_prob >= 0.75:
            return 'HIGH'
        elif max_prob >= 0.65:
            return 'MEDIUM'
        return 'LOW'


class MockRealtimePredictor:
    """Mock realtime predictor for testing."""

    def __init__(self):
        self._tickers = ['AAPL', 'GOOGL', 'MSFT']

    def predict(self, ticker, include_all_models=False, include_features=False):
        if ticker.upper() not in [t.upper() for t in self._tickers]:
            raise ValueError(f"No trained models for {ticker}")
        return MockPredictionResult(ticker.upper())

    def predict_batch(self, tickers, include_all_models=False, include_features=False):
        results = {}
        for ticker in tickers:
            try:
                results[ticker.upper()] = self.predict(ticker)
            except ValueError:
                pass
        return results

    def get_top_opportunities(self, tickers, direction='up', min_probability=0.7, top_n=10):
        results = []
        for ticker in tickers[:top_n]:
            try:
                result = self.predict(ticker)
                if direction == 'up' and result.up_probability >= min_probability:
                    results.append((ticker, result))
                elif direction == 'down' and result.down_probability >= min_probability:
                    results.append((ticker, result))
            except ValueError:
                pass
        return results

    def get_prediction_stats(self, ticker):
        return {
            'accuracy': 0.75,
            'total_predictions': 100,
            'avg_probability': 0.72
        }


@pytest.fixture
def mock_model_manager():
    """Provide a mock model manager."""
    return MockModelManager()


@pytest.fixture
def mock_predictor():
    """Provide a mock realtime predictor."""
    return MockRealtimePredictor()


@pytest.fixture
def mock_db_session():
    """Provide a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def test_client(mock_model_manager, mock_predictor, mock_db_session):
    """Create a test client with mocked dependencies."""
    from src.api.main import app
    from src.api import dependencies

    # Override dependencies
    app.dependency_overrides[dependencies.get_model_manager] = lambda: mock_model_manager
    app.dependency_overrides[dependencies.get_realtime_predictor] = lambda: mock_predictor
    app.dependency_overrides[dependencies.get_db] = lambda: mock_db_session

    with TestClient(app) as client:
        yield client

    # Clear overrides after test
    app.dependency_overrides.clear()


@pytest.fixture
def mock_market_status():
    """Provide mock market status data."""
    return {
        'is_open': False,
        'is_trading_day': True,
        'current_time_et': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_open': '09:30',
        'market_close': '16:00',
        'reason': 'After hours'
    }
