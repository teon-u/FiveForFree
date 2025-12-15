"""Test model implementations."""
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.models.base_model import BaseModel


class MockModel(BaseModel):
    """Mock model for testing."""

    def train(self, X, y, X_val=None, y_val=None):
        self.is_trained = True

    def predict_proba(self, X):
        return np.random.rand(len(X))

    def save(self, path: Path):
        pass

    def load(self, path: Path):
        self.is_trained = True


def test_base_model_prediction_tracking():
    """Test that base model tracks predictions correctly."""
    model = MockModel(ticker="TEST", target="up", model_type="xgboost")

    # Make a prediction
    X_test = np.random.rand(10, 57)
    probs = model.predict_proba(X_test)

    # Track prediction with timestamp
    timestamp = datetime.now()
    model.record_prediction(probs[0], timestamp)

    assert len(model.prediction_history) == 1
    assert model.prediction_history[0]["probability"] == probs[0]
    assert model.prediction_history[0]["timestamp"] == timestamp


def test_base_model_accuracy_calculation():
    """Test accuracy calculation with outcomes."""
    model = MockModel(ticker="TEST", target="up", model_type="xgboost")

    # Record predictions with timestamps
    now = datetime.now()
    ts1 = now - timedelta(hours=1)
    ts2 = now - timedelta(hours=2)

    model.record_prediction(0.8, ts1)  # Predicted positive (>= 0.5)
    model.record_prediction(0.3, ts2)  # Predicted negative (< 0.5)

    # Update outcomes
    model.update_outcome(ts1, True)   # Prediction was correct (predicted positive, actual positive)
    model.update_outcome(ts2, False)  # Prediction was correct (predicted negative, actual negative)

    accuracy = model.get_recent_accuracy(hours=24)
    assert accuracy == 1.0  # Both predictions correct
