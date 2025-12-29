"""Model manager for handling multiple models per ticker."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

from loguru import logger
from sqlalchemy import select

from config.settings import settings
from src.models.base_model import BaseModel
from src.utils.database import get_db, ModelPerformance


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
                        # Check if 'model' attribute exists and has a value, while '_model' is None
                        if hasattr(model, 'model') and model.model is not None:
                            if not hasattr(model, '_model') or model._model is None:
                                model._model = model.model
                                logger.debug(f"Migrated legacy 'model' to '_model' for {model_file.name}")

                        # For LSTM/Transformer models, call load() to restore PyTorch weights from .pt file
                        # For XGBoost models, call load() to restore booster from .json file
                        if model_type in ('lstm', 'transformer', 'xgboost'):
                            try:
                                model.load(model_file)
                                logger.debug(f"Loaded model weights for {model_file.name}")
                            except Exception as e:
                                logger.warning(f"Could not load model weights for {model_file.name}: {e}")

                        # For Ensemble models, set the model_manager reference
                        # (it was excluded from pickle to avoid 340MB files)
                        if model_type == 'ensemble':
                            model.set_model_manager(self)

                        # Ensure cached_stats attributes exist (backward compatibility)
                        if not hasattr(model, 'cached_stats'):
                            model.cached_stats = None
                        if not hasattr(model, 'cached_stats_updated_at'):
                            model.cached_stats_updated_at = None

                        # Restore cached_stats from DB if not present in memory
                        if model.cached_stats is None:
                            db_stats = self._load_cached_stats_from_db(ticker, model_type, target)
                            if db_stats:
                                model.cached_stats = db_stats
                                model.cached_stats_updated_at = datetime.now()
                                logger.debug(f"Restored cached_stats from DB for {ticker}/{model_type}_{target}")

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

        # Call model's save() first to save PyTorch weights (.pt files) for LSTM/Transformer
        model.save(model_path)

        # Then pickle the model object (metadata and non-PyTorch state)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Saved model to {model_path}")
        return model_path

    def save_models(self, ticker: str) -> int:
        """
        Save all models for a ticker to disk.

        Args:
            ticker: Ticker symbol

        Returns:
            Number of models saved.
        """
        ticker = ticker.upper()
        saved_count = 0

        if ticker not in self._models:
            logger.warning(f"No models found for ticker {ticker}")
            return 0

        for model_type, targets in self._models[ticker].items():
            for target, model in targets.items():
                try:
                    self.save_model(ticker, model_type, target, model)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Failed to save {model_type}_{target} for {ticker}: {e}")

        logger.info(f"Saved {saved_count} models for {ticker}")
        return saved_count

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

    def get_all_models(self, ticker: str, target: str) -> Dict[str, BaseModel]:
        """
        Get all models for a ticker and target.

        Args:
            ticker: Ticker symbol
            target: Target direction (up, down, volatility, direction)

        Returns:
            Dict with structure {model_type: model}
        """
        ticker = ticker.upper()
        result: Dict[str, BaseModel] = {}

        if ticker not in self._models:
            return result

        for model_type in self._models[ticker]:
            if target in self._models[ticker][model_type]:
                result[model_type] = self._models[ticker][model_type][target]

        return result

    def get_model_performances(self, ticker: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get performance metrics for all models of a ticker.

        Uses database ModelPerformance table as primary source.
        Falls back to in-memory stats if no DB record exists.

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with structure {target: {model_type: metrics}}
        """
        ticker = ticker.upper()
        performances: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if ticker not in self._models:
            return performances

        # Query DB for this ticker's performance records
        # Store extracted values (not SQLAlchemy objects) to avoid detached instance errors
        db_performances: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {target: {model_type: extracted_data}}
        try:
            with get_db() as db:
                # Get the most recent performance record for each model_type and symbol
                stmt = (
                    select(ModelPerformance)
                    .where(ModelPerformance.symbol == ticker)
                    .order_by(ModelPerformance.evaluation_date.desc())
                )
                records = db.execute(stmt).scalars().all()

                # Group by target direction (from metrics_json) and model_type
                # Prefer records with precision > 0 over empty records
                for record in records:
                    target_direction = None
                    metrics_json = record.metrics_json
                    if metrics_json and 'target_direction' in metrics_json:
                        target_direction = metrics_json['target_direction']

                    if target_direction is None:
                        continue

                    if target_direction not in db_performances:
                        db_performances[target_direction] = {}

                    # Check if this record has meaningful precision data
                    precision = metrics_json.get('precision', 0.0) if metrics_json else 0.0
                    has_data = precision > 0

                    # Extract values from SQLAlchemy object while session is still active
                    extracted_data = {
                        'metrics_json': dict(metrics_json) if metrics_json else {},
                        'total_trades': record.total_trades,
                        'total_predictions': record.total_predictions,
                        'precision': precision
                    }

                    # Prefer records with precision > 0
                    if record.model_type not in db_performances[target_direction]:
                        # No record yet, use this one
                        db_performances[target_direction][record.model_type] = extracted_data
                    else:
                        # Already have a record, only replace if:
                        # - Current record has no data (precision=0) AND new record has data
                        existing_data = db_performances[target_direction][record.model_type]
                        existing_precision = existing_data.get('precision', 0.0)
                        existing_has_data = existing_precision > 0

                        if not existing_has_data and has_data:
                            # Replace empty record with one that has data
                            db_performances[target_direction][record.model_type] = extracted_data

        except Exception as e:
            logger.warning(f"Failed to query DB for performances: {e}")

        # Build performance dict from models
        for model_type, targets in self._models[ticker].items():
            for target, model in targets.items():
                if target not in performances:
                    performances[target] = {}

                # Try in-memory stats first (most recent realtime data)
                stats = model.get_prediction_stats(hours=settings.BACKTEST_HOURS)
                db_data = db_performances.get(target, {}).get(model_type)

                if stats['total_predictions'] > 0:
                    # Use in-memory stats (most recent - reflects realtime collection)
                    performances[target][model_type] = {
                        'hit_rate_50h': stats['precision'] * 100,  # Precision as percentage
                        'precision': stats['precision'],
                        'recall': stats['recall'],
                        'signal_rate': stats.get('signal_rate', 0.0),
                        'signal_count': stats.get('signal_count', 0),
                        'practicality_grade': stats.get('practicality_grade', 'D'),
                        'total_predictions': stats['total_predictions'],
                        'is_trained': model.is_trained,
                        'last_trained': model.last_trained.isoformat() if hasattr(model, 'last_trained') and model.last_trained else None,
                        'source': 'memory'  # Indicate data source for debugging
                    }
                elif db_data and db_data.get('metrics_json'):
                    # Fallback to DB values (historical data when no recent predictions)
                    metrics_json = db_data['metrics_json']
                    precision = metrics_json.get('precision', 0.0)
                    recall = metrics_json.get('recall', 0.0)
                    prediction_rate = metrics_json.get('prediction_rate', 0.0)

                    performances[target][model_type] = {
                        'hit_rate_50h': precision * 100,  # Precision as percentage
                        'precision': precision,
                        'recall': recall,
                        'signal_rate': prediction_rate,
                        'signal_count': db_data['total_trades'],
                        'practicality_grade': self._calculate_practicality_grade(precision, prediction_rate),
                        'total_predictions': db_data['total_predictions'],
                        'is_trained': model.is_trained,
                        'last_trained': model.last_trained.isoformat() if hasattr(model, 'last_trained') and model.last_trained else None,
                        'source': 'db'  # Indicate data source for debugging
                    }
                else:
                    # Option D: Fallback priority for initial metrics
                    # 3순위: initial_precision (validation 기반)
                    # 4순위: train_accuracy (최후 fallback)
                    initial_precision = getattr(model, 'initial_precision', 0.0)
                    initial_recall = getattr(model, 'initial_recall', 0.0)
                    train_accuracy = getattr(model, 'train_accuracy', 0.0)
                    validation_samples = getattr(model, 'validation_samples', 0)

                    if initial_precision > 0:
                        # 3순위: Use validation metrics
                        precision = initial_precision
                        recall = initial_recall
                        source = 'validation'
                        total_predictions = validation_samples
                    elif train_accuracy > 0:
                        # 4순위: Use training accuracy as fallback
                        precision = train_accuracy
                        recall = 0.0  # No recall info from train_accuracy alone
                        source = 'training'
                        total_predictions = getattr(model, 'training_samples', 0)
                    else:
                        # No data available at all
                        precision = 0.0
                        recall = 0.0
                        source = 'none'
                        total_predictions = 0

                    performances[target][model_type] = {
                        'hit_rate_50h': precision * 100,  # Precision as percentage
                        'precision': precision,
                        'recall': recall,
                        'signal_rate': 0.0,  # No signal rate from initial metrics
                        'signal_count': 0,
                        'practicality_grade': self._calculate_practicality_grade(precision, 0.0),
                        'total_predictions': total_predictions,
                        'is_trained': model.is_trained,
                        'last_trained': model.last_trained.isoformat() if hasattr(model, 'last_trained') and model.last_trained else None,
                        'source': source  # 'validation', 'training', or 'none'
                    }

        return performances

    def _calculate_practicality_grade(self, precision: float, signal_rate: float) -> str:
        """
        Calculate practicality grade based on precision and signal rate.

        Args:
            precision: Prediction precision (0-1)
            signal_rate: Signal rate (0-1)

        Returns:
            Grade from A to D
        """
        if precision >= 0.7 and signal_rate >= 0.1:
            return 'A'
        elif precision >= 0.6 and signal_rate >= 0.05:
            return 'B'
        elif precision >= 0.5 and signal_rate >= 0.02:
            return 'C'
        else:
            return 'D'

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

    def _load_cached_stats_from_db(self, ticker: str, model_type: str, target: str) -> Optional[dict]:
        """
        Load cached performance stats from DB for a specific model.

        Args:
            ticker: Ticker symbol
            model_type: Model type (xgboost, lightgbm, etc.)
            target: Target direction (up/down)

        Returns:
            Dict with cached stats or None if not found
        """
        try:
            with get_db() as db:
                # Get the most recent performance record for this model
                stmt = (
                    select(ModelPerformance)
                    .where(ModelPerformance.symbol == ticker)
                    .where(ModelPerformance.model_type == model_type)
                    .order_by(ModelPerformance.evaluation_date.desc())
                    .limit(1)
                )
                record = db.execute(stmt).scalar_one_or_none()

                if record and record.metrics_json:
                    metrics = record.metrics_json
                    # Only return if target matches and has meaningful data
                    if metrics.get('target_direction') == target:
                        precision = metrics.get('precision', 0.0)
                        if precision > 0:
                            return {
                                'precision': precision,
                                'recall': metrics.get('recall', 0.0),
                                'accuracy': record.accuracy,
                                'signal_rate': metrics.get('signal_rate', 0.0),
                                'signal_count': metrics.get('signal_count', 0),
                                'total_predictions': record.total_predictions,
                                'true_positives': metrics.get('true_positives', 0),
                                'false_positives': metrics.get('false_positives', 0),
                                'true_negatives': metrics.get('true_negatives', 0),
                                'false_negatives': metrics.get('false_negatives', 0),
                                'avg_probability': metrics.get('avg_probability', 0.5),
                            }
        except Exception as e:
            logger.warning(f"Failed to load cached stats from DB for {ticker}/{model_type}_{target}: {e}")

        return None

    def save_cached_stats_to_db(self, ticker: str, model_type: str, target: str, stats: dict) -> bool:
        """
        Save cached performance stats to DB for persistence across restarts.

        Args:
            ticker: Ticker symbol
            model_type: Model type
            target: Target direction
            stats: Performance stats dict

        Returns:
            True if saved successfully
        """
        try:
            with get_db() as db:
                now = datetime.now()
                record = ModelPerformance(
                    model_type=model_type,
                    symbol=ticker,
                    evaluation_date=now,
                    period_start=now,
                    period_end=now,
                    total_predictions=stats.get('total_predictions', 0),
                    correct_predictions=int(stats.get('accuracy', 0) * stats.get('total_predictions', 0)),
                    accuracy=stats.get('accuracy', 0.0),
                    metrics_json={
                        'target_direction': target,
                        'precision': stats.get('precision', 0.0),
                        'recall': stats.get('recall', 0.0),
                        'signal_rate': stats.get('signal_rate', 0.0),
                        'signal_count': stats.get('signal_count', 0),
                        'true_positives': stats.get('true_positives', 0),
                        'false_positives': stats.get('false_positives', 0),
                        'true_negatives': stats.get('true_negatives', 0),
                        'false_negatives': stats.get('false_negatives', 0),
                        'avg_probability': stats.get('avg_probability', 0.5),
                    }
                )
                db.add(record)
                db.commit()
                logger.debug(f"Saved cached stats to DB for {ticker}/{model_type}_{target}")
                return True
        except Exception as e:
            logger.error(f"Failed to save cached stats to DB: {e}")
            return False

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

    def refresh_all_ensemble_weights(self) -> Dict[str, bool]:
        """
        Refresh weights for all ensemble models across all tickers.

        Called periodically by scheduler to update ensemble weights
        based on latest prediction_history performance.

        Returns:
            Dict with {ticker_target: weights_changed}
        """
        results: Dict[str, bool] = {}

        for ticker in self._tickers:
            for target in ['up', 'down']:
                key = f"{ticker}_{target}"
                try:
                    if ticker in self._models and 'ensemble' in self._models[ticker]:
                        if target in self._models[ticker]['ensemble']:
                            ensemble = self._models[ticker]['ensemble'][target]
                            changed = ensemble.refresh_weights()
                            results[key] = changed
                except Exception as e:
                    logger.warning(f"Failed to refresh ensemble weights for {key}: {e}")
                    results[key] = False

        # Log summary
        refreshed = sum(1 for v in results.values() if v)
        if refreshed > 0:
            logger.info(f"Refreshed ensemble weights: {refreshed}/{len(results)} updated")

        return results
