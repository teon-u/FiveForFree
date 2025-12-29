"""LightGBM model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np

from loguru import logger

from src.models.base_model import BaseModel
from config.settings import settings

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. LightGBMModel will not work.")

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


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

        # GPU 사용 시도 (LightGBM GPU 빌드 필요)
        use_gpu = settings.USE_GPU and HAS_CUDA

        # 클래스 가중치 계산 (클래스 불균형 해결)
        y_array = np.asarray(y)
        n_pos = (y_array == 1).sum()
        n_neg = (y_array == 0).sum()
        scale_pos_weight = min(n_neg / max(n_pos, 1), settings.MAX_CLASS_WEIGHT)

        logger.info(f"LightGBM scale_pos_weight: {scale_pos_weight:.2f} "
                    f"(pos={n_pos}, neg={n_neg})")

        # GPU 파라미터 설정 (LightGBM 4.0+)
        gpu_params = {}
        if use_gpu:
            gpu_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
            }

        try:
            self._model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=scale_pos_weight,  # 클래스 불균형 가중치
                random_state=42,
                verbose=-1,
                **gpu_params
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

            if use_gpu:
                logger.debug("LightGBM: Using GPU")

        except Exception as e:
            # GPU 실패 시 CPU로 fallback
            if use_gpu:
                logger.warning(f"LightGBM GPU failed, falling back to CPU: {e}")
                self._model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    scale_pos_weight=scale_pos_weight,  # 클래스 불균형 가중치
                    random_state=42,
                    verbose=-1
                )

                X_train = np.asarray(X)
                y_train = np.asarray(y)

                eval_set = None
                if X_val is not None and y_val is not None:
                    eval_set = [(np.asarray(X_val), np.asarray(y_val))]

                self._model.fit(
                    X_train, y_train,
                    eval_set=eval_set
                )
            else:
                raise

        self.is_trained = True
        self._update_last_trained()

        # Calculate and store train accuracy using validation set
        if X_val is not None and y_val is not None:
            y_pred = (self.predict_proba(np.asarray(X_val)) >= 0.5).astype(int)
            y_val_arr = np.asarray(y_val)
            self.train_accuracy = float(np.mean(y_pred == y_val_arr))
            logger.info(f"LightGBM model trained for {self.ticker} {self.target} (val_acc={self.train_accuracy:.2%})")
        else:
            # Fallback: use training accuracy
            X_train_np = np.asarray(X)
            y_train_np = np.asarray(y)
            y_pred = (self.predict_proba(X_train_np) >= 0.5).astype(int)
            self.train_accuracy = float(np.mean(y_pred == y_train_np))
            logger.info(f"LightGBM model trained for {self.ticker} {self.target} (train_acc={self.train_accuracy:.2%})")

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
