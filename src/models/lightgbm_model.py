"""LightGBM model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np

from loguru import logger

from src.models.base_model import BaseModel

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. LightGBMModel will not work.")


class LightGBMModel(BaseModel):
    """LightGBM-based prediction model."""

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
        Initialize LightGBM model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__(ticker=ticker, target=target, model_type='lightgbm')

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model: Optional[lgb.LGBMClassifier] = None

    def train(self, X, y, X_val=None, y_val=None):
        """Train the LightGBM model."""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed")

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=-1
        )

        # Convert to numpy arrays to avoid feature name warnings
        X_train = np.asarray(X)
        y_train = np.asarray(y)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.asarray(X_val), np.asarray(y_val))]

        self._model.fit(
            X_train, y_train,
            eval_set=eval_set
        )

        self.is_trained = True
        self._update_last_trained()
        logger.info(f"LightGBM model trained for {self.ticker} {self.target}")

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
            self._model.booster_.save_model(str(path.with_suffix('.txt')))
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        txt_path = path.with_suffix('.txt')
        if txt_path.exists() and HAS_LIGHTGBM:
            self._model = lgb.LGBMClassifier()
            self._model._Booster = lgb.Booster(model_file=str(txt_path))
