"""Model manager for handling multiple models per ticker."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

from loguru import logger

from config.settings import settings
from src.models.base_model import BaseModel


class ModelManager:
    """
    Manages multiple models for each ticker.

    Handles model creation, loading, saving, and selection of best performing models.
    """

    MODEL_TYPES = ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']
    TARGETS = ['up', 'down']

    def __init__(self, models_path: Optional[Path] = None):
        """
        Initialize ModelManager.

        Args:
            models_path: Path to models directory. Defaults to data/models.
        """
        self.models_path = models_path or Path(settings.DATA_DIR) / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Dict[ticker][model_type][target] = model
        self._models: Dict[str, Dict[str, Dict[str, BaseModel]]] = {}
        self._tickers: List[str] = []

        logger.info(f"ModelManager initialized with path: {self.models_path}")

    def get_tickers(self) -> List[str]:
        """Get list of tickers with models."""
        return list(self._tickers)

    def load_all_models(self) -> int:
        """
        Load all saved models from disk.

        Returns:
            Number of models loaded.
        """
        loaded_count = 0

        if not self.models_path.exists():
            logger.warning(f"Models path does not exist: {self.models_path}")
            return 0

        # Find all ticker directories
        for ticker_dir in self.models_path.iterdir():
            if not ticker_dir.is_dir():
                continue

            ticker = ticker_dir.name.upper()

            if ticker not in self._models:
                self._models[ticker] = {}
                self._tickers.append(ticker)

            # Load models for this ticker
            for model_file in ticker_dir.glob("*.pkl"):
                try:
                    # Parse filename: {target}_{model_type}.pkl (e.g., down_lightgbm.pkl)
                    parts = model_file.stem.split('_')
                    if len(parts) >= 2:
                        target = parts[0]      # First part is target (up/down)
                        model_type = parts[1]  # Second part is model type

                        if model_type not in self._models[ticker]:
                            self._models[ticker][model_type] = {}

                        # Load the model
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)

                        # Handle legacy pickle files that saved as 'model' instead of '_model'
                        if hasattr(model, 'model') and not hasattr(model, '_model'):
                            model._model = model.model

                        self._models[ticker][model_type][target] = model
                        loaded_count += 1
                        logger.debug(f"Loaded {model_type}_{target} for {ticker}")

                except Exception as e:
                    logger.error(f"Failed to load model {model_file}: {e}")

        logger.info(f"Loaded {loaded_count} models for {len(self._tickers)} tickers")
        return loaded_count

    def save_model(self, ticker: str, model_type: str, target: str, model: BaseModel) -> Path:
        """
        Save a model to disk.

        Args:
            ticker: Ticker symbol
            model_type: Model type (xgboost, lightgbm, etc.)
            target: Target direction (up, down)
            model: Model instance to save

        Returns:
            Path to saved model file.
        """
        ticker = ticker.upper()
        ticker_dir = self.models_path / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        # Use {target}_{model_type}.pkl to match existing file naming convention
        model_path = ticker_dir / f"{target}_{model_type}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Saved model to {model_path}")
        return model_path

    def get_or_create_model(
        self,
        ticker: str,
        model_type: str,
        target: str
    ) -> Tuple[str, BaseModel]:
        """
        Get existing model or create new one.

        Args:
            ticker: Ticker symbol
            model_type: Model type
            target: Target direction

        Returns:
            Tuple of (model_type, model instance)
        """
        ticker = ticker.upper()

        # Initialize structures if needed
        if ticker not in self._models:
            self._models[ticker] = {}
            if ticker not in self._tickers:
                self._tickers.append(ticker)

        if model_type not in self._models[ticker]:
            self._models[ticker][model_type] = {}

        # Return existing model if available
        if target in self._models[ticker][model_type]:
            return model_type, self._models[ticker][model_type][target]

        # Create new model
        model = self._create_model(ticker, model_type, target)
        self._models[ticker][model_type][target] = model

        return model_type, model

    def _create_model(self, ticker: str, model_type: str, target: str) -> BaseModel:
        """
        Create a new model instance.

        Args:
            ticker: Ticker symbol
            model_type: Model type
            target: Target direction

        Returns:
            New model instance
        """
        # Import model classes dynamically to avoid circular imports
        if model_type == 'xgboost':
            from src.models.xgboost_model import XGBoostModel
            return XGBoostModel(ticker=ticker, target=target)
        elif model_type == 'lightgbm':
            from src.models.lightgbm_model import LightGBMModel
            return LightGBMModel(ticker=ticker, target=target)
        elif model_type == 'lstm':
            from src.models.lstm_model import LSTMModel
            return LSTMModel(ticker=ticker, target=target)
        elif model_type == 'transformer':
            from src.models.transformer_model import TransformerModel
            return TransformerModel(ticker=ticker, target=target)
        elif model_type == 'ensemble':
            from src.models.ensemble_model import EnsembleModel
            return EnsembleModel(ticker=ticker, target=target, model_manager=self)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_best_model(self, ticker: str, target: str) -> Tuple[str, BaseModel]:
        """
        Get the best performing model for a ticker and target.

        Based on 50-hour hit rate performance.

        Args:
            ticker: Ticker symbol
            target: Target direction (up, down)

        Returns:
            Tuple of (model_type, best model instance)

        Raises:
            ValueError: If no trained models found
        """
        ticker = ticker.upper()

        if ticker not in self._models:
            raise ValueError(f"No models found for ticker {ticker}")

        best_model = None
        best_model_type = None
        best_hit_rate = -1.0

        for model_type, targets in self._models[ticker].items():
            if target in targets:
                model = targets[target]
                if model.is_trained:
                    hit_rate = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)
                    if hit_rate > best_hit_rate:
                        best_hit_rate = hit_rate
                        best_model = model
                        best_model_type = model_type

        if best_model is None:
            raise ValueError(f"No trained models found for {ticker} {target}")

        return best_model_type, best_model

    def get_model_performances(self, ticker: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get performance metrics for all models of a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with structure {target: {model_type: metrics}}
        """
        ticker = ticker.upper()
        performances: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if ticker not in self._models:
            return performances

        for model_type, targets in self._models[ticker].items():
            for target, model in targets.items():
                if target not in performances:
                    performances[target] = {}

                stats = model.get_prediction_stats(hours=settings.BACKTEST_HOURS)

                performances[target][model_type] = {
                    'hit_rate_50h': stats['precision'] * 100,  # Precision as percentage
                    'precision': stats['precision'],
                    'recall': stats['recall'],
                    'signal_rate': stats.get('signal_rate', 0.0),
                    'signal_count': stats.get('signal_count', 0),
                    'practicality_grade': stats.get('practicality_grade', 'D'),
                    'total_predictions': stats['total_predictions'],
                    'is_trained': model.is_trained,
                    'last_trained': model.last_trained.isoformat() if hasattr(model, 'last_trained') and model.last_trained else None
                }

        return performances

    def validate_models(self, ticker: str) -> Dict[str, bool]:
        """
        Validate which models are trained for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with {model_type_target: is_trained}
        """
        ticker = ticker.upper()
        validation: Dict[str, bool] = {}

        for model_type in self.MODEL_TYPES:
            for target in self.TARGETS:
                key = f"{model_type}_{target}"

                if ticker in self._models:
                    if model_type in self._models[ticker]:
                        if target in self._models[ticker][model_type]:
                            validation[key] = self._models[ticker][model_type][target].is_trained
                            continue

                validation[key] = False

        return validation

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all models.

        Returns:
            Summary dict with counts and statistics.
        """
        total_models = 0
        trained_models = 0

        for ticker in self._models:
            for model_type in self._models[ticker]:
                for target, model in self._models[ticker][model_type].items():
                    total_models += 1
                    if model.is_trained:
                        trained_models += 1

        return {
            'total_tickers': len(self._tickers),
            'tickers': list(self._tickers),
            'total_models': total_models,
            'trained_models': trained_models,
            'untrained_models': total_models - trained_models,
            'models_path': str(self.models_path)
        }

    def train_all_models(
        self,
        ticker: str,
        X_train,
        y_train_up,
        y_train_down,
        X_val=None,
        y_val_up=None,
        y_val_down=None
    ) -> Dict[str, bool]:
        """
        Train all models for a ticker.

        Args:
            ticker: Ticker symbol
            X_train: Training features
            y_train_up: Up labels
            y_train_down: Down labels
            X_val: Validation features (optional)
            y_val_up: Up validation labels (optional)
            y_val_down: Down validation labels (optional)

        Returns:
            Dict with {model_type_target: success}
        """
        results: Dict[str, bool] = {}

        for model_type in self.MODEL_TYPES:
            if model_type == 'ensemble':
                continue  # Train ensemble after base models

            for target, y_train, y_val in [
                ('up', y_train_up, y_val_up),
                ('down', y_train_down, y_val_down)
            ]:
                key = f"{model_type}_{target}"

                try:
                    _, model = self.get_or_create_model(ticker, model_type, target)
                    model.train(X_train, y_train, X_val, y_val)
                    self.save_model(ticker, model_type, target, model)
                    results[key] = True
                    logger.info(f"Trained {key} for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to train {key} for {ticker}: {e}")
                    results[key] = False

        # Train ensemble models after base models
        for target, y_train, y_val in [
            ('up', y_train_up, y_val_up),
            ('down', y_train_down, y_val_down)
        ]:
            key = f"ensemble_{target}"

            try:
                _, model = self.get_or_create_model(ticker, 'ensemble', target)
                model.train(X_train, y_train, X_val, y_val)
                self.save_model(ticker, 'ensemble', target, model)
                results[key] = True
                logger.info(f"Trained {key} for {ticker}")
            except Exception as e:
                logger.error(f"Failed to train {key} for {ticker}: {e}")
                results[key] = False

        return results
