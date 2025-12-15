"""
Incremental Learning Module for NASDAQ Prediction System

Enables efficient online learning for tree-based models:
- Continuous adaptation to new market data
- Efficient updates without full retraining
- Maintains model performance over time
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger

from src.models.model_manager import ModelManager
from config.settings import settings


class IncrementalTrainer:
    """
    Incremental learning trainer for continuous model updates.

    Optimized for tree-based models (XGBoost, LightGBM) which support
    efficient incremental training via boosting continuation.

    Features:
    - Hourly incremental updates with new market data
    - Automatic data buffering and batching
    - Selective training (only tree models by default)
    - Performance monitoring and rollback capability
    """

    def __init__(
        self,
        model_manager: ModelManager,
        min_samples_for_update: int = 50,
        max_buffer_size: int = 1000
    ):
        """
        Initialize incremental trainer.

        Args:
            model_manager: ModelManager instance for model coordination
            min_samples_for_update: Minimum new samples required before update
            max_buffer_size: Maximum samples to keep in buffer
        """
        self.model_manager = model_manager
        self.min_samples = min_samples_for_update
        self.max_buffer_size = max_buffer_size

        # Data buffers for each ticker: {ticker: {'X': [], 'y_up': [], 'y_down': [], 'timestamps': []}}
        self.data_buffers: Dict[str, Dict[str, List]] = {}

        # Track incremental training events
        self.training_log: List[Dict[str, Any]] = []

        # Performance tracking before/after updates
        self.performance_history: Dict[str, List[Dict]] = {}

        logger.info(
            f"IncrementalTrainer initialized "
            f"(min_samples={min_samples_for_update}, max_buffer={max_buffer_size})"
        )

    def add_sample(
        self,
        ticker: str,
        X: np.ndarray,
        y_up: float,
        y_down: float,
        timestamp: datetime = None
    ) -> None:
        """
        Add a new training sample to the buffer.

        Args:
            ticker: Stock ticker symbol
            X: Feature vector (1D array)
            y_up: Label for upward movement (0 or 1)
            y_down: Label for downward movement (0 or 1)
            timestamp: When this sample was created (default: now)
        """
        if ticker not in self.data_buffers:
            self.data_buffers[ticker] = {
                'X': [],
                'y_up': [],
                'y_down': [],
                'timestamps': []
            }

        buffer = self.data_buffers[ticker]

        # Add new sample
        buffer['X'].append(X)
        buffer['y_up'].append(y_up)
        buffer['y_down'].append(y_down)
        buffer['timestamps'].append(timestamp or datetime.now())

        # Maintain max buffer size (FIFO)
        if len(buffer['X']) > self.max_buffer_size:
            buffer['X'].pop(0)
            buffer['y_up'].pop(0)
            buffer['y_down'].pop(0)
            buffer['timestamps'].pop(0)

    def add_batch(
        self,
        ticker: str,
        X: np.ndarray,
        y_up: np.ndarray,
        y_down: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> None:
        """
        Add a batch of training samples to the buffer.

        Args:
            ticker: Stock ticker symbol
            X: Feature matrix (n_samples, n_features)
            y_up: Labels for upward movement
            y_down: Labels for downward movement
            timestamps: List of timestamps for each sample
        """
        n_samples = len(X)

        if timestamps is None:
            timestamps = [datetime.now()] * n_samples

        for i in range(n_samples):
            self.add_sample(ticker, X[i], y_up[i], y_down[i], timestamps[i])

        logger.debug(f"Added {n_samples} samples to {ticker} buffer")

    def update_models(
        self,
        ticker: str,
        targets: List[str] = None,
        model_types: List[str] = None,
        force: bool = False
    ) -> Dict[str, bool]:
        """
        Incrementally update models for a ticker if sufficient new data is available.

        Args:
            ticker: Stock ticker symbol
            targets: List of targets to update (default: ["up", "down"])
            model_types: List of model types to update (default: tree models only)
            force: Force update even if min_samples not reached

        Returns:
            Dictionary mapping model keys to update success status
        """
        targets = targets or settings.PREDICTION_TARGETS
        model_types = model_types or ["xgboost", "lightgbm"]  # Only tree models by default

        # Check if ticker has buffered data
        if ticker not in self.data_buffers:
            logger.debug(f"No buffered data for {ticker}")
            return {}

        buffer = self.data_buffers[ticker]
        n_samples = len(buffer['X'])

        # Check minimum samples requirement
        if n_samples < self.min_samples and not force:
            logger.debug(
                f"Insufficient samples for {ticker}: {n_samples} < {self.min_samples}"
            )
            return {}

        logger.info(f"Incrementally updating models for {ticker} with {n_samples} new samples")

        # Convert buffer to arrays
        X_new = np.array(buffer['X'])
        y_up_new = np.array(buffer['y_up'])
        y_down_new = np.array(buffer['y_down'])

        results = {}
        start_time = datetime.now()

        # Update models for each target
        for target in targets:
            y_new = y_up_new if target == "up" else y_down_new

            for model_type in model_types:
                model_key = f"{model_type}_{target}"

                try:
                    # Get existing model
                    model = self.model_manager.get_or_create_model(ticker, model_type, target)

                    if not model.is_trained:
                        logger.warning(
                            f"Model {model_key} for {ticker} not trained yet, skipping incremental update"
                        )
                        results[model_key] = False
                        continue

                    # Record performance before update
                    accuracy_before = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)

                    # Incremental training
                    model.incremental_train(X_new, y_new)

                    # Record performance after update
                    accuracy_after = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)

                    # Track performance change
                    self._record_performance_change(
                        ticker, model_type, target,
                        accuracy_before, accuracy_after,
                        n_samples
                    )

                    results[model_key] = True
                    logger.info(
                        f"Updated {model_key} for {ticker}: "
                        f"accuracy {accuracy_before:.3f} -> {accuracy_after:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to update {model_key} for {ticker}: {e}")
                    results[model_key] = False

        # Log training event
        elapsed = (datetime.now() - start_time).total_seconds()
        self._log_training_event(ticker, targets, model_types, n_samples, elapsed, results)

        # Clear buffer after successful update
        if any(results.values()):
            self.clear_buffer(ticker)
            logger.debug(f"Cleared buffer for {ticker}")

        return results

    def update_all_tickers(
        self,
        targets: List[str] = None,
        model_types: List[str] = None,
        min_samples_override: Optional[int] = None
    ) -> Dict[str, Dict[str, bool]]:
        """
        Update models for all tickers with buffered data.

        Args:
            targets: List of targets to update (default: ["up", "down"])
            model_types: List of model types to update (default: tree models)
            min_samples_override: Override min_samples requirement

        Returns:
            Dictionary mapping ticker to model update results
        """
        logger.info(f"Updating models for all tickers with buffered data")

        # Temporarily override min_samples if specified
        original_min_samples = self.min_samples
        if min_samples_override is not None:
            self.min_samples = min_samples_override

        results = {}
        tickers_to_update = list(self.data_buffers.keys())

        for ticker in tickers_to_update:
            ticker_results = self.update_models(ticker, targets, model_types)
            if ticker_results:
                results[ticker] = ticker_results

        # Restore original min_samples
        self.min_samples = original_min_samples

        logger.info(f"Updated models for {len(results)} tickers")

        return results

    def get_buffer_status(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of data buffers.

        Args:
            ticker: Optional specific ticker (default: all tickers)

        Returns:
            Dictionary with buffer status information
        """
        if ticker:
            if ticker not in self.data_buffers:
                return {
                    'ticker': ticker,
                    'samples': 0,
                    'ready_for_update': False
                }

            buffer = self.data_buffers[ticker]
            n_samples = len(buffer['X'])

            return {
                'ticker': ticker,
                'samples': n_samples,
                'ready_for_update': n_samples >= self.min_samples,
                'oldest_sample': buffer['timestamps'][0] if buffer['timestamps'] else None,
                'newest_sample': buffer['timestamps'][-1] if buffer['timestamps'] else None
            }
        else:
            # Return status for all tickers
            status = {}
            for tkr in self.data_buffers:
                status[tkr] = self.get_buffer_status(tkr)

            return {
                'total_tickers': len(self.data_buffers),
                'ready_for_update': sum(1 for s in status.values() if s['ready_for_update']),
                'total_samples': sum(s['samples'] for s in status.values()),
                'tickers': status
            }

    def clear_buffer(self, ticker: str) -> None:
        """
        Clear data buffer for a ticker.

        Args:
            ticker: Stock ticker symbol
        """
        if ticker in self.data_buffers:
            del self.data_buffers[ticker]
            logger.debug(f"Cleared buffer for {ticker}")

    def clear_all_buffers(self) -> None:
        """Clear all data buffers."""
        n_tickers = len(self.data_buffers)
        self.data_buffers.clear()
        logger.info(f"Cleared all buffers ({n_tickers} tickers)")

    def cleanup_old_samples(self, days: int = 7) -> None:
        """
        Remove samples older than specified days from all buffers.

        Args:
            days: Number of days to keep
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        cleaned_count = 0

        for ticker, buffer in self.data_buffers.items():
            # Find index of first sample after cutoff
            timestamps = buffer['timestamps']
            keep_indices = [i for i, ts in enumerate(timestamps) if ts >= cutoff_time]

            if len(keep_indices) < len(timestamps):
                # Keep only recent samples
                buffer['X'] = [buffer['X'][i] for i in keep_indices]
                buffer['y_up'] = [buffer['y_up'][i] for i in keep_indices]
                buffer['y_down'] = [buffer['y_down'][i] for i in keep_indices]
                buffer['timestamps'] = [buffer['timestamps'][i] for i in keep_indices]

                removed = len(timestamps) - len(keep_indices)
                cleaned_count += removed
                logger.debug(f"Removed {removed} old samples from {ticker} buffer")

        logger.info(f"Cleaned up {cleaned_count} samples older than {days} days")

    def _record_performance_change(
        self,
        ticker: str,
        model_type: str,
        target: str,
        accuracy_before: float,
        accuracy_after: float,
        n_samples: int
    ) -> None:
        """
        Record performance change from incremental training.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            target: "up" or "down"
            accuracy_before: Accuracy before update
            accuracy_after: Accuracy after update
            n_samples: Number of samples used for update
        """
        key = f"{ticker}_{model_type}_{target}"

        if key not in self.performance_history:
            self.performance_history[key] = []

        self.performance_history[key].append({
            'timestamp': datetime.now(),
            'accuracy_before': accuracy_before,
            'accuracy_after': accuracy_after,
            'improvement': accuracy_after - accuracy_before,
            'n_samples': n_samples
        })

    def _log_training_event(
        self,
        ticker: str,
        targets: List[str],
        model_types: List[str],
        n_samples: int,
        elapsed_seconds: float,
        results: Dict[str, bool]
    ) -> None:
        """
        Log an incremental training event.

        Args:
            ticker: Stock ticker
            targets: Targets that were updated
            model_types: Model types that were updated
            n_samples: Number of samples used
            elapsed_seconds: Time taken
            results: Training results
        """
        self.training_log.append({
            'timestamp': datetime.now(),
            'ticker': ticker,
            'targets': targets,
            'model_types': model_types,
            'n_samples': n_samples,
            'elapsed_seconds': elapsed_seconds,
            'success_count': sum(1 for v in results.values() if v),
            'total_count': len(results),
            'results': results
        })

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of incremental training activities.

        Returns:
            Dictionary with training statistics
        """
        if not self.training_log:
            return {
                'total_updates': 0,
                'tickers_updated': 0,
                'total_samples_processed': 0
            }

        total_updates = len(self.training_log)
        tickers_updated = set(event['ticker'] for event in self.training_log)
        total_samples = sum(event['n_samples'] for event in self.training_log)
        total_time = sum(event['elapsed_seconds'] for event in self.training_log)

        # Recent performance improvements
        recent_improvements = []
        for key, history in self.performance_history.items():
            if history:
                latest = history[-1]
                recent_improvements.append({
                    'model': key,
                    'improvement': latest['improvement'],
                    'timestamp': latest['timestamp']
                })

        # Sort by improvement
        recent_improvements.sort(key=lambda x: x['improvement'], reverse=True)

        return {
            'total_updates': total_updates,
            'tickers_updated': len(tickers_updated),
            'total_samples_processed': total_samples,
            'total_time_seconds': total_time,
            'avg_time_per_update': total_time / total_updates if total_updates > 0 else 0,
            'top_improvements': recent_improvements[:10],
            'recent_events': self.training_log[-10:]
        }

    def get_performance_history(
        self,
        ticker: str,
        model_type: str,
        target: str
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific model.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            target: "up" or "down"

        Returns:
            List of performance change records
        """
        key = f"{ticker}_{model_type}_{target}"
        return self.performance_history.get(key, [])

    def export_buffer_to_dataframe(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Export buffer data to pandas DataFrame.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with buffered data, or None if no buffer exists
        """
        if ticker not in self.data_buffers:
            return None

        buffer = self.data_buffers[ticker]

        if not buffer['X']:
            return None

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': buffer['timestamps'],
            'y_up': buffer['y_up'],
            'y_down': buffer['y_down']
        })

        # Add feature columns
        X_array = np.array(buffer['X'])
        for i in range(X_array.shape[1]):
            df[f'feature_{i}'] = X_array[:, i]

        return df
