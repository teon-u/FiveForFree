"""XGBoost model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np

from loguru import logger

from src.models.base_model import BaseModel
from config.settings import settings

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. XGBoostModel will not work.")

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


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

        # GPU 사용 여부 결정
        use_gpu = settings.USE_GPU and HAS_CUDA

        # 클래스 가중치 계산 (클래스 불균형 해결)
        y_array = np.asarray(y)
        n_pos = (y_array == 1).sum()
        n_neg = (y_array == 0).sum()
        scale_pos_weight = min(n_neg / max(n_pos, 1), settings.MAX_CLASS_WEIGHT)

        logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f} "
                    f"(pos={n_pos}, neg={n_neg})")

        # XGBoost 2.0+: tree_method='hist' + device='cuda' (gpu_hist는 deprecated)
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric='logloss',
            tree_method='hist',  # hist는 device에 따라 GPU/CPU 자동 선택
            device='cuda' if use_gpu else 'cpu',  # XGBoost 2.0+ GPU device
            scale_pos_weight=scale_pos_weight,  # 클래스 불균형 가중치
            random_state=42
        )

        if use_gpu:
            logger.debug("XGBoost: Using GPU (device=cuda)")

        # Convert to numpy arrays to avoid feature name warnings
        X_train = np.asarray(X)
        y_train = np.asarray(y)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.asarray(X_val), np.asarray(y_val))]

        self._model.fit(
            X_train, y_train,
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

    def __getstate__(self):
        """Customize pickle serialization to exclude _model."""
        state = self.__dict__.copy()
        # _model은 별도 JSON 파일로 저장되므로 pickle에서 제외
        # XGBClassifier는 _estimator_type 미정의로 pickle 실패 방지
        state['_model'] = None
        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        # _model은 load() 메서드에서 JSON 파일로부터 복원됨

    def save(self, path: Path):
        """Save model to disk."""
        if self._model is not None:
            # XGBoost 3.x에서 sklearn wrapper의 save_model()은 _estimator_type 필요
            # get_booster()를 사용하여 native booster로 저장 (LightGBM과 동일한 패턴)
            self._model.get_booster().save_model(str(path.with_suffix('.json')))
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        json_path = path.with_suffix('.json')
        if json_path.exists() and HAS_XGBOOST:
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(json_path))
            # Set to CPU for prediction (avoids GPU transfer overhead for small batches)
            self._model.set_params(device='cpu')
