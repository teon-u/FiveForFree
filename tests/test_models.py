"""Test model implementations."""
import pytest
import numpy as np
from pathlib import Path
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
    model = MockModel("TEST", "xgboost", "up")

    # Make a prediction
    X_test = np.random.rand(10, 57)
    probs = model.predict_proba(X_test)

    # Track prediction
    model.record_prediction(probs[0], "TEST_ENTRY_ID")

    assert len(model.prediction_history) == 1
    assert model.prediction_history[0]["predicted_probability"] == probs[0]


def test_base_model_accuracy_calculation():
    """Test accuracy calculation with outcomes."""
    model = MockModel("TEST", "xgboost", "up")

    # Record predictions with outcomes
    model.record_prediction(0.8, "entry1")
    model.record_prediction(0.3, "entry2")

    model.update_prediction_outcome("entry1", True)  # Correct
    model.update_prediction_outcome("entry2", False)  # Correct

    accuracy = model.get_recent_accuracy(hours=24)
    assert accuracy == 1.0  # Both predictions correct
