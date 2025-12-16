"""Ensemble model for NASDAQ prediction."""

from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

from loguru import logger

from src.models.base_model import BaseModel

if TYPE_CHECKING:
    from src.models.model_manager import ModelManager

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. EnsembleModel meta-learner will use simple averaging.")


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from base models.

    Uses a meta-learner (Logistic Regression) to combine base model predictions.
    """

    BASE_MODEL_TYPES = ['xgboost', 'lightgbm', 'lstm', 'transformer']

    def __init__(
        self,
        ticker: str,
        target: str,
        model_manager: Optional['ModelManager'] = None,
        **kwargs
    ):
        """
        Initialize Ensemble model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            model_manager: ModelManager instance for accessing base models
        """
        super().__init__(ticker=ticker, target=target, model_type='ensemble')

        self._model_manager = model_manager
        self._meta_learner: Optional[LogisticRegression] = None
        self._base_model_weights: Dict[str, float] = {}
        self._trained_base_models: List[str] = []

    def set_model_manager(self, model_manager: 'ModelManager'):
        """Set the model manager for accessing base models."""
        self._model_manager = model_manager

    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the ensemble model.

        Collects predictions from base models and trains a meta-learner.
        """
        if self._model_manager is None:
            raise ValueError("ModelManager not set. Call set_model_manager() first.")

        # Collect base model predictions
        base_predictions = []
        self._trained_base_models = []

        for model_type in self.BASE_MODEL_TYPES:
            try:
                _, base_model = self._model_manager.get_or_create_model(
                    self.ticker, model_type, self.target
                )

                if base_model.is_trained:
                    preds = base_model.predict_proba(X)
                    base_predictions.append(preds)
                    self._trained_base_models.append(model_type)
                    logger.debug(f"Added predictions from {model_type}")

            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_type}: {e}")

        if len(base_predictions) == 0:
            logger.warning("No base models available, using simple model")
            # Fallback: train a simple logistic regression on raw features
            if HAS_SKLEARN:
                self._meta_learner = LogisticRegression(max_iter=1000, random_state=42)
                self._meta_learner.fit(X, y)
            self.is_trained = True
            self._update_last_trained()
            return

        # Stack base predictions
        X_meta = np.column_stack(base_predictions)

        # Train meta-learner
        if HAS_SKLEARN:
            self._meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            self._meta_learner.fit(X_meta, y)

            # Extract weights from coefficients
            coefs = self._meta_learner.coef_[0]
            total = np.sum(np.abs(coefs))
            if total > 0:
                for i, model_type in enumerate(self._trained_base_models):
                    self._base_model_weights[model_type] = abs(coefs[i]) / total
            else:
                # Equal weights if coefficients are zero
                for model_type in self._trained_base_models:
                    self._base_model_weights[model_type] = 1.0 / len(self._trained_base_models)
        else:
            # Simple averaging without sklearn
            for model_type in self._trained_base_models:
                self._base_model_weights[model_type] = 1.0 / len(self._trained_base_models)

        self.is_trained = True
        self._update_last_trained()
        logger.info(f"Ensemble model trained for {self.ticker} {self.target} with {len(self._trained_base_models)} base models")

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using ensemble."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self._model_manager is None:
            raise ValueError("ModelManager not set")

        # Collect base model predictions
        base_predictions = []

        for model_type in self._trained_base_models:
            try:
                _, base_model = self._model_manager.get_or_create_model(
                    self.ticker, model_type, self.target
                )

                if base_model.is_trained:
                    preds = base_model.predict_proba(X)
                    base_predictions.append(preds)

            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_type}: {e}")

        if len(base_predictions) == 0:
            # Fallback to meta-learner on raw features
            if self._meta_learner is not None:
                proba = self._meta_learner.predict_proba(X)
                return proba[:, 1] if proba.ndim > 1 else proba
            raise ValueError("No predictions available")

        # Use meta-learner if available
        X_meta = np.column_stack(base_predictions)

        if self._meta_learner is not None:
            proba = self._meta_learner.predict_proba(X_meta)
            return proba[:, 1] if proba.ndim > 1 else proba

        # Simple weighted average fallback
        weighted_sum = np.zeros(len(X))
        total_weight = 0

        for i, model_type in enumerate(self._trained_base_models):
            weight = self._base_model_weights.get(model_type, 1.0)
            weighted_sum += base_predictions[i] * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else weighted_sum

    def get_base_model_weights(self) -> Dict[str, float]:
        """Get weights assigned to each base model."""
        return self._base_model_weights.copy()

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble model statistics."""
        return {
            'meta_learner_type': 'LogisticRegression' if self._meta_learner is not None else 'WeightedAverage',
            'trained_base_models': len(self._trained_base_models),
            'total_base_models': len(self.BASE_MODEL_TYPES),
            'base_models': self._trained_base_models,
            'weights': self._base_model_weights
        }

    def save(self, path: Path):
        """Save model to disk."""
        # Meta-learner will be saved via pickle in parent class
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        # Note: model_manager must be set again after loading
