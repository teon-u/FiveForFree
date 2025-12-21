"""
Multi-Strategy Ensemble Model for NASDAQ Prediction.

Implements three ensemble strategies:
1. Precision-Weighted Voting: Weights predictions by model precision
2. Stacking with XGBoost: Uses XGBoost as meta-learner
3. Dynamic Model Selection: Selects best model based on recent performance
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Tuple
from enum import Enum
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

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Stacking will use LogisticRegression.")


class EnsembleStrategy(Enum):
    """Available ensemble strategies."""
    PRECISION_WEIGHTED = "precision_weighted"  # Weight by precision
    STACKING = "stacking"  # Meta-learner stacking
    DYNAMIC_SELECTION = "dynamic_selection"  # Best model selection
    HYBRID = "hybrid"  # Combine all strategies


class EnsembleModel(BaseModel):
    """
    Multi-Strategy Ensemble Model.

    Combines predictions from base models using multiple strategies:
    - Precision-Weighted Voting: Higher precision = higher weight
    - Stacking: XGBoost/LogisticRegression meta-learner
    - Dynamic Selection: Pick best performing model
    - Hybrid: Weighted combination of all strategies
    """

    BASE_MODEL_TYPES = ['xgboost', 'lightgbm', 'lstm', 'transformer']

    # Breakeven precision for 3:1 R/R strategy
    BREAKEVEN_PRECISION = 0.30

    def __init__(
        self,
        ticker: str,
        target: str,
        model_manager: Optional['ModelManager'] = None,
        strategy: EnsembleStrategy = EnsembleStrategy.HYBRID,
        min_precision_threshold: float = 0.25,
        **kwargs
    ):
        """
        Initialize Ensemble model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            model_manager: ModelManager instance for accessing base models
            strategy: Ensemble strategy to use
            min_precision_threshold: Minimum precision to include model
        """
        super().__init__(ticker=ticker, target=target, model_type='ensemble')

        self._model_manager = model_manager
        self._strategy = strategy
        self._min_precision_threshold = min_precision_threshold

        # Strategy-specific learners
        self._stacking_learner: Optional[Any] = None  # XGBoost or LogisticRegression

        # Model weights and metrics
        self._precision_weights: Dict[str, float] = {}
        self._base_model_precisions: Dict[str, float] = {}
        self._trained_base_models: List[str] = []
        self._best_model: Optional[str] = None

        # Strategy weights for hybrid
        self._strategy_weights = {
            'precision_weighted': 0.4,
            'stacking': 0.4,
            'dynamic_selection': 0.2
        }

    def set_model_manager(self, model_manager: 'ModelManager'):
        """Set the model manager for accessing base models."""
        self._model_manager = model_manager

    def _get_model_precisions(self) -> Dict[str, float]:
        """Get precision for each base model from prediction history."""
        precisions = {}

        if self._model_manager is None:
            return precisions

        for model_type in self.BASE_MODEL_TYPES:
            try:
                _, base_model = self._model_manager.get_or_create_model(
                    self.ticker, model_type, self.target
                )

                if base_model.is_trained:
                    # Get precision from model's prediction history
                    precision = base_model.get_precision_at_threshold(
                        threshold=0.7,
                        hours=50
                    )
                    if precision is not None and precision > 0:
                        precisions[model_type] = precision
                        logger.debug(f"{model_type} precision: {precision:.2%}")
                    else:
                        # Default precision if no history
                        precisions[model_type] = 0.35

            except Exception as e:
                logger.warning(f"Failed to get precision for {model_type}: {e}")
                precisions[model_type] = 0.35  # Default

        return precisions

    def _calculate_precision_weights(self) -> Dict[str, float]:
        """Calculate weights based on precision above breakeven."""
        weights = {}

        # Get excess precision above breakeven
        excess_precisions = {}
        for model_type, precision in self._base_model_precisions.items():
            if precision >= self._min_precision_threshold:
                excess = max(0, precision - self.BREAKEVEN_PRECISION)
                excess_precisions[model_type] = excess

        # Normalize to get weights
        total_excess = sum(excess_precisions.values())
        if total_excess > 0:
            for model_type, excess in excess_precisions.items():
                weights[model_type] = excess / total_excess
        else:
            # Equal weights if no excess precision
            n_models = len(self._base_model_precisions)
            if n_models > 0:
                for model_type in self._base_model_precisions:
                    weights[model_type] = 1.0 / n_models

        return weights

    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the ensemble model.

        Trains all strategies:
        1. Precision-weighted: Calculate weights from precision history
        2. Stacking: Train XGBoost/LR on base model predictions
        3. Dynamic Selection: Identify best performing model
        """
        if self._model_manager is None:
            raise ValueError("ModelManager not set. Call set_model_manager() first.")

        # Get precision for each base model
        self._base_model_precisions = self._get_model_precisions()

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

                    # Handle sequence models returning fewer predictions
                    if len(preds) < len(y):
                        # Pad with NaN and mask later
                        padded_preds = np.full(len(y), np.nan)
                        padded_preds[-len(preds):] = preds
                        preds = padded_preds

                    base_predictions.append(preds)
                    self._trained_base_models.append(model_type)
                    logger.debug(f"Added predictions from {model_type}")

            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_type}: {e}")

        if len(base_predictions) == 0:
            logger.warning("No base models available for ensemble")
            self.is_trained = True
            self._update_last_trained()
            return

        # 1. Calculate precision-based weights
        self._precision_weights = self._calculate_precision_weights()
        logger.info(f"Precision weights: {self._precision_weights}")

        # 2. Find best model for dynamic selection
        if self._base_model_precisions:
            self._best_model = max(
                self._base_model_precisions.items(),
                key=lambda x: x[1]
            )[0]
            logger.info(f"Best model (by precision): {self._best_model}")

        # 3. Train stacking meta-learner
        X_meta = np.column_stack(base_predictions)

        # Create mask for valid predictions (handle sequence model padding)
        valid_mask = ~np.isnan(X_meta).any(axis=1)
        X_meta_valid = X_meta[valid_mask]
        y_valid = y[valid_mask] if hasattr(y, '__getitem__') else np.array(y)[valid_mask]

        if len(X_meta_valid) > 10:  # Need minimum samples
            # Ensure numpy arrays to avoid feature name warnings
            X_meta_train = np.asarray(X_meta_valid)
            y_meta_train = np.asarray(y_valid)

            if HAS_XGBOOST:
                # 레이블 검증: 모든 값이 동일하면 메타러너 학습 스킵
                positive_ratio = float(np.mean(y_meta_train))
                if positive_ratio == 0.0 or positive_ratio == 1.0:
                    logger.warning(
                        f"Skipping XGBoost meta-learner for {self.ticker} {self.target}: "
                        f"all labels are {int(positive_ratio)}"
                    )
                else:
                    # Use XGBoost as meta-learner with clipped base_score
                    self._stacking_learner = xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=3,
                        learning_rate=0.1,
                        objective='binary:logistic',
                        eval_metric='logloss',
                        random_state=42,
                        base_score=np.clip(positive_ratio, 0.01, 0.99)
                    )
                    self._stacking_learner.fit(X_meta_train, y_meta_train)
                    logger.info("Trained XGBoost meta-learner")
            elif HAS_SKLEARN:
                # Fallback to LogisticRegression
                self._stacking_learner = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
                self._stacking_learner.fit(X_meta_train, y_meta_train)
                logger.info("Trained LogisticRegression meta-learner")

        self.is_trained = True
        self._update_last_trained()
        logger.info(
            f"Ensemble trained for {self.ticker} {self.target} "
            f"with {len(self._trained_base_models)} base models, "
            f"strategy: {self._strategy.value}"
        )

    def _predict_precision_weighted(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using precision-weighted voting."""
        weighted_sum = None
        total_weight = 0

        for model_type, preds in base_predictions.items():
            weight = self._precision_weights.get(model_type, 0)
            if weight > 0:
                if weighted_sum is None:
                    weighted_sum = preds * weight
                else:
                    weighted_sum += preds * weight
                total_weight += weight

        if weighted_sum is None or total_weight == 0:
            # Fallback to simple average
            return np.mean(list(base_predictions.values()), axis=0)

        return weighted_sum / total_weight

    def _predict_stacking(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using stacking meta-learner."""
        if self._stacking_learner is None:
            # Fallback to simple average
            return np.mean(list(base_predictions.values()), axis=0)

        # Stack predictions in same order as training
        X_meta = np.column_stack([
            base_predictions[model_type]
            for model_type in self._trained_base_models
            if model_type in base_predictions
        ])

        proba = self._stacking_learner.predict_proba(X_meta)
        return proba[:, 1] if proba.ndim > 1 else proba

    def _predict_dynamic_selection(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using best performing model."""
        if self._best_model and self._best_model in base_predictions:
            return base_predictions[self._best_model]

        # Fallback to model with highest precision in current predictions
        if self._base_model_precisions:
            for model_type in sorted(
                self._base_model_precisions.keys(),
                key=lambda x: self._base_model_precisions[x],
                reverse=True
            ):
                if model_type in base_predictions:
                    return base_predictions[model_type]

        # Last resort: average
        return np.mean(list(base_predictions.values()), axis=0)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict probabilities using ensemble.

        Combines predictions based on selected strategy.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        if self._model_manager is None:
            raise ValueError("ModelManager not set")

        # Collect base model predictions
        base_predictions = {}

        for model_type in self._trained_base_models:
            try:
                _, base_model = self._model_manager.get_or_create_model(
                    self.ticker, model_type, self.target
                )

                if base_model.is_trained:
                    preds = base_model.predict_proba(X)
                    # Ensure predictions are always 1D arrays (not scalars)
                    preds = np.atleast_1d(preds)
                    base_predictions[model_type] = preds

            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_type}: {e}")

        if len(base_predictions) == 0:
            raise ValueError("No predictions available from base models")

        # Ensure all predictions have same length
        min_len = min(len(p) for p in base_predictions.values())
        base_predictions = {k: v[-min_len:] for k, v in base_predictions.items()}

        # Apply strategy
        if self._strategy == EnsembleStrategy.PRECISION_WEIGHTED:
            return self._predict_precision_weighted(base_predictions)

        elif self._strategy == EnsembleStrategy.STACKING:
            return self._predict_stacking(base_predictions)

        elif self._strategy == EnsembleStrategy.DYNAMIC_SELECTION:
            return self._predict_dynamic_selection(base_predictions)

        elif self._strategy == EnsembleStrategy.HYBRID:
            # Combine all strategies
            pred_pw = self._predict_precision_weighted(base_predictions)
            pred_stack = self._predict_stacking(base_predictions)
            pred_dyn = self._predict_dynamic_selection(base_predictions)

            # Weighted average of strategies
            w = self._strategy_weights
            return (
                w['precision_weighted'] * pred_pw +
                w['stacking'] * pred_stack +
                w['dynamic_selection'] * pred_dyn
            )

        # Default fallback
        return np.mean(list(base_predictions.values()), axis=0)

    def get_base_model_weights(self) -> Dict[str, float]:
        """Get precision-based weights assigned to each base model."""
        return self._precision_weights.copy()

    def get_base_model_precisions(self) -> Dict[str, float]:
        """Get precision values for each base model."""
        return self._base_model_precisions.copy()

    def get_best_model(self) -> Optional[str]:
        """Get the best performing base model."""
        return self._best_model

    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble model statistics."""
        stacking_type = 'None'
        if self._stacking_learner is not None:
            if HAS_XGBOOST and isinstance(self._stacking_learner, xgb.XGBClassifier):
                stacking_type = 'XGBoost'
            elif HAS_SKLEARN:
                stacking_type = 'LogisticRegression'

        return {
            'strategy': self._strategy.value,
            'stacking_learner': stacking_type,
            'trained_base_models': len(self._trained_base_models),
            'total_base_models': len(self.BASE_MODEL_TYPES),
            'base_models': self._trained_base_models,
            'precision_weights': self._precision_weights,
            'base_precisions': self._base_model_precisions,
            'best_model': self._best_model,
            'strategy_weights': self._strategy_weights if self._strategy == EnsembleStrategy.HYBRID else None
        }

    def set_strategy(self, strategy: EnsembleStrategy):
        """Change the ensemble strategy."""
        self._strategy = strategy
        logger.info(f"Ensemble strategy changed to: {strategy.value}")

    def set_strategy_weights(self, weights: Dict[str, float]):
        """Set weights for hybrid strategy."""
        total = sum(weights.values())
        if total > 0:
            self._strategy_weights = {k: v/total for k, v in weights.items()}
            logger.info(f"Strategy weights updated: {self._strategy_weights}")

    def __getstate__(self):
        """Customize pickle serialization to exclude _model_manager."""
        state = self.__dict__.copy()
        # _model_manager contains ALL models for ALL tickers
        # Storing it would cause massive pickle files (340MB each!)
        # It will be restored via set_model_manager() after loading
        state['_model_manager'] = None
        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        # _model_manager is None after unpickling
        # ModelManager.load_all_models() will call set_model_manager()

    def save(self, path: Path):
        """Save model to disk."""
        # Meta-learner and weights will be saved via pickle in parent class
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        # Note: model_manager must be set again after loading
