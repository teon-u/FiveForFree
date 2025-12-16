"""XGBoost model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np

from loguru import logger

from src.models.base_model import BaseModel

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. XGBoostModel will not work.")


class XGBoostModel(BaseModel):
    """XGBoost-based prediction model."""

    def __init__(
        self,
        ticker: str,
        target: str,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize XGBoost model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__(ticker=ticker, target=target, model_type='xgboost')

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[xgb.XGBClassifier] = None

    def train(self, X, y, X_val=None, y_val=None):
        """Train the XGBoost model."""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed")

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',  # GPU-friendly
            random_state=42
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )

        self.is_trained = True
        self._update_last_trained()
        logger.info(f"XGBoost model trained for {self.ticker} {self.target}")

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self._model is None:
            raise ValueError("Model not trained")

        proba = self._model.predict_proba(X)
        # Return probability of positive class
        return proba[:, 1] if proba.ndim > 1 else proba

    def save(self, path: Path):
        """Save model to disk."""
        if self._model is not None:
            self._model.save_model(str(path.with_suffix('.json')))
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        json_path = path.with_suffix('.json')
        if json_path.exists() and HAS_XGBOOST:
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(json_path))
