"""Abstract base class for all prediction models."""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import pickle
from loguru import logger

from config.settings import settings


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    All models must implement:
    - train(): Train the model
    - predict_proba(): Generate probability predictions
    - save(): Persist model to disk
    - load(): Load model from disk

    Provides tracking of:
    - Training status
    - Recent predictions vs actual outcomes
    - 50-hour rolling accuracy
    """

    def __init__(self, ticker: str, target: str = "up", model_type: str = "base"):
        """
        Initialize base model.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            target: Prediction target, either "up" or "down"
            model_type: Type of model (xgboost, lightgbm, lstm, transformer, ensemble)
        """
        self.ticker = ticker
        self.target = target
        self.model_type = model_type
        self.is_trained = False
        self.model = None

        # Track predictions and outcomes for accuracy calculation
        self.prediction_history: list[dict] = []

        # Cached performance stats (persisted for server restart)
        # These are used when prediction_history is empty or stale
        self.cached_stats: Optional[dict] = None
        self.cached_stats_updated_at: Optional[datetime] = None

        # Model metadata
        self.created_at = datetime.now()
        self.last_trained_at: Optional[datetime] = None
        self.training_samples: int = 0
        self.train_accuracy: float = 0.0  # Training accuracy as fallback for precision

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the model on provided data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for positive class.

        Args:
            X: Features to predict on (n_samples, n_features)

        Returns:
            Array of probabilities (n_samples,)
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path to load model from
        """
        pass

    def _update_last_trained(self, n_samples: Optional[int] = None) -> None:
        """
        Update the last trained timestamp and optionally sample count.

        Should be called at the end of train() in all child classes.

        Args:
            n_samples: Number of samples used for training (optional)
        """
        self.last_trained_at = datetime.now()
        if n_samples is not None:
            self.training_samples = n_samples
        logger.debug(f"{self.model_type} model for {self.ticker}/{self.target} trained at {self.last_trained_at}")

    def incremental_train(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """
        Incrementally train model with new data.

        Default implementation retrains from scratch.
        Tree models should override this for efficient incremental learning.

        Args:
            X_new: New training features
            y_new: New training labels
        """
        logger.warning(f"{self.model_type} does not support true incremental training, retraining from scratch")
        self.train(X_new, y_new)

    def record_prediction(self, probability: float, timestamp: datetime,
                         features: Optional[np.ndarray] = None) -> None:
        """
        Record a prediction for later accuracy tracking.

        Args:
            probability: Predicted probability
            timestamp: When prediction was made
            features: Optional features used for prediction
        """
        self.prediction_history.append({
            'probability': probability,
            'timestamp': timestamp,
            'actual_outcome': None,  # Will be updated later
            'features': features
        })

    def update_outcome(self, timestamp: datetime, actual_outcome: bool) -> None:
        """
        Update the actual outcome for a previous prediction.

        Args:
            timestamp: Timestamp of the prediction to update
            actual_outcome: True if target was reached, False otherwise
        """
        for pred in self.prediction_history:
            if pred['timestamp'] == timestamp and pred['actual_outcome'] is None:
                pred['actual_outcome'] = actual_outcome
                break

    def get_recent_accuracy(self, hours: int = 50) -> float:
        """
        Calculate accuracy over recent predictions within specified hours.

        This is the key metric used by ModelManager to select best model.

        Args:
            hours: Number of hours to look back

        Returns:
            Accuracy as a float between 0 and 1, or 0 if no data
        """
        if not self.prediction_history:
            return 0.0

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter to recent predictions with known outcomes
        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if not recent:
            return 0.0

        # Calculate accuracy based on probability threshold
        threshold = settings.PROBABILITY_THRESHOLD
        correct = 0
        for pred in recent:
            predicted_positive = pred['probability'] >= threshold
            if predicted_positive == pred['actual_outcome']:
                correct += 1

        return correct / len(recent)

    def get_precision_at_threshold(self, hours: int = 50, threshold: float = 0.5) -> float:
        """
        Calculate precision for predictions above threshold.

        Precision = TP / (TP + FP)
        When model predicts positive (prob >= threshold), how often is it correct?

        This is a more meaningful metric than overall accuracy because:
        - It measures reliability of HIGH confidence predictions
        - It's not skewed by class imbalance

        Args:
            hours: Number of hours to look back
            threshold: Probability threshold for positive prediction

        Returns:
            Precision as a float between 0 and 1, or train_accuracy as fallback
        """
        # Use train_accuracy as fallback when no prediction history
        if not self.prediction_history:
            return self.train_accuracy

        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        # Use train_accuracy as fallback when no recent predictions
        if not recent:
            return self.train_accuracy

        # Count predictions where model predicted positive (prob >= threshold)
        true_positives = 0
        false_positives = 0

        for pred in recent:
            if pred['probability'] >= threshold:
                if pred['actual_outcome']:
                    true_positives += 1
                else:
                    false_positives += 1

        total_positive_predictions = true_positives + false_positives

        if total_positive_predictions == 0:
            return 0.0

        return true_positives / total_positive_predictions

    def get_signal_metrics(self, hours: int = 50, threshold: float = 0.5) -> dict:
        """
        Get comprehensive signal metrics for trading decisions.

        Returns metrics that are meaningful for trading:
        - precision: When model says "buy", how often is it right?
        - recall: Of all actual opportunities, how many did we catch?
        - signal_count: How many signals were generated?
        - true_positive_rate: Same as recall

        Args:
            hours: Number of hours to look back
            threshold: Probability threshold for positive prediction

        Returns:
            Dictionary with signal metrics
        """
        if not self.prediction_history:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'signal_count': 0,
                'true_positive_count': 0,
                'actual_positive_count': 0
            }

        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if not recent:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'signal_count': 0,
                'true_positive_count': 0,
                'actual_positive_count': 0
            }

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        actual_positives = 0

        for pred in recent:
            if pred['actual_outcome']:
                actual_positives += 1

            if pred['probability'] >= threshold:
                if pred['actual_outcome']:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if pred['actual_outcome']:
                    false_negatives += 1

        signal_count = true_positives + false_positives
        precision = true_positives / signal_count if signal_count > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'signal_count': signal_count,
            'true_positive_count': true_positives,
            'actual_positive_count': actual_positives
        }

    def cache_performance_stats(self) -> None:
        """
        Cache current performance stats for persistence across server restarts.

        This should be called after backtest or when prediction_history has been updated.
        The cached stats will be used when prediction_history is empty or stale.
        """
        # Calculate stats from current prediction_history
        stats = self._calculate_prediction_stats_internal()

        if stats['total_predictions'] > 0:
            self.cached_stats = stats
            self.cached_stats_updated_at = datetime.now()
            logger.debug(f"Cached stats for {self.ticker}/{self.target}: precision={stats['precision']:.2%}")

    def get_prediction_stats(self, hours: int = 50) -> dict[str, Any]:
        """
        Get detailed statistics about recent predictions.

        Uses cached stats when prediction_history is empty or stale.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with statistics
        """
        # First try to get stats from recent predictions
        stats = self._calculate_prediction_stats_internal(hours)

        # If no recent predictions, use cached stats
        if stats['total_predictions'] == 0 and self.cached_stats is not None:
            # Return cached stats with a flag indicating they are cached
            cached = self.cached_stats.copy()
            cached['is_cached'] = True
            cached['cached_at'] = self.cached_stats_updated_at.isoformat() if self.cached_stats_updated_at else None
            return cached

        return stats

    def _calculate_prediction_stats_internal(self, hours: int = 50) -> dict[str, Any]:
        """
        Internal method to calculate prediction stats from prediction_history.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if not recent:
            return {
                'total_predictions': 0,
                'accuracy': 0.0,
                'avg_probability': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'signal_rate': 0.0,
                'signal_count': 0,
                'practicality_grade': 'D',
                'bias_warning': None,
                'is_cached': False
            }

        # Calculate confusion matrix using configurable threshold
        threshold = settings.PROBABILITY_THRESHOLD
        tp = sum(1 for p in recent if p['probability'] >= threshold and p['actual_outcome'])
        fp = sum(1 for p in recent if p['probability'] >= threshold and not p['actual_outcome'])
        tn = sum(1 for p in recent if p['probability'] < threshold and not p['actual_outcome'])
        fn = sum(1 for p in recent if p['probability'] < threshold and p['actual_outcome'])

        accuracy = (tp + tn) / len(recent) if recent else 0.0
        avg_prob = np.mean([p['probability'] for p in recent])

        # Signal Rate = (predictions with prob >= threshold) / total opportunities
        signals = tp + fp  # Number of buy signals
        total = len(recent)
        signal_rate = signals / total if total > 0 else 0.0

        # Precision (key metric for profitability)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Practicality Grade based on precision and signal_rate
        # A: precision >= 50% AND signal_rate >= 10%
        # B: precision >= 30% AND signal_rate >= 10%
        # C: precision >= 30% AND signal_rate < 10%
        # D: precision < 30%
        if precision >= 0.50 and signal_rate >= 0.10:
            practicality_grade = 'A'
        elif precision >= 0.30 and signal_rate >= 0.10:
            practicality_grade = 'B'
        elif precision >= 0.30:
            practicality_grade = 'C'
        else:
            practicality_grade = 'D'

        # Bias warning detection
        bias_warning = None
        if signal_rate < 0.05 or signal_rate > 0.95:
            bias_warning = "Model biased to one direction"
        elif accuracy > 0.80 and precision < 0.20:
            bias_warning = "High accuracy but low precision - class imbalance"

        return {
            'total_predictions': total,
            'accuracy': accuracy,
            'avg_probability': float(avg_prob),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'signal_rate': signal_rate,
            'signal_count': signals,
            'practicality_grade': practicality_grade,
            'bias_warning': bias_warning,
            'is_cached': False
        }

    def cleanup_old_predictions(self, days: int = 7) -> None:
        """
        Remove prediction history older than specified days to save memory.

        Args:
            days: Number of days to keep
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        self.prediction_history = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time
        ]

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model metadata and status.

        Returns:
            Dictionary with model information
        """
        return {
            'ticker': self.ticker,
            'target': self.target,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'created_at': self.created_at.isoformat(),
            'last_trained_at': self.last_trained_at.isoformat() if self.last_trained_at else None,
            'training_samples': self.training_samples,
            'total_predictions': len(self.prediction_history)
        }

    def save_metadata(self, path: Path) -> None:
        """
        Save model metadata separately from model weights.

        Args:
            path: Directory to save metadata
        """
        metadata = {
            'ticker': self.ticker,
            'target': self.target,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'created_at': self.created_at,
            'last_trained_at': self.last_trained_at,
            'training_samples': self.training_samples,
            'prediction_history': self.prediction_history
        }

        metadata_file = path / f"{self.ticker}_{self.target}_{self.model_type}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved metadata to {metadata_file}")

    def load_metadata(self, path: Path) -> None:
        """
        Load model metadata from disk.

        Args:
            path: Directory to load metadata from
        """
        metadata_file = path / f"{self.ticker}_{self.target}_{self.model_type}_metadata.pkl"

        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        self.ticker = metadata['ticker']
        self.target = metadata['target']
        self.model_type = metadata['model_type']
        self.is_trained = metadata['is_trained']
        self.created_at = metadata['created_at']
        self.last_trained_at = metadata['last_trained_at']
        self.training_samples = metadata['training_samples']
        self.prediction_history = metadata['prediction_history']

        logger.info(f"Loaded metadata from {metadata_file}")

    def get_roc_curve_data(self, hours: int = 50) -> dict[str, Any]:
        """
        Calculate ROC curve data for model evaluation.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with FPR, TPR, thresholds, and AUC
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if len(recent) < 10:
            return {
                'fpr': [],
                'tpr': [],
                'thresholds': [],
                'auc': 0.0,
                'insufficient_data': True,
                'sample_count': len(recent)
            }

        # Extract probabilities and labels
        y_true = np.array([1 if p['actual_outcome'] else 0 for p in recent])
        y_scores = np.array([p['probability'] for p in recent])

        # Calculate ROC curve points
        thresholds = np.linspace(0, 1, 101)
        fpr_list = []
        tpr_list = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            # Calculate TP, FP, TN, FN
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(fpr_list) - 1):
            auc += (fpr_list[i+1] - fpr_list[i]) * (tpr_list[i] + tpr_list[i+1]) / 2

        return {
            'fpr': [float(x) for x in fpr_list],
            'tpr': [float(x) for x in tpr_list],
            'thresholds': [float(x) for x in thresholds],
            'auc': abs(float(auc)),
            'insufficient_data': False,
            'sample_count': len(recent)
        }

    def get_precision_recall_curve(self, hours: int = 50) -> dict[str, Any]:
        """
        Calculate Precision-Recall curve data.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with precision, recall, thresholds, and average precision
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if len(recent) < 10:
            return {
                'precision': [],
                'recall': [],
                'thresholds': [],
                'average_precision': 0.0,
                'insufficient_data': True,
                'sample_count': len(recent)
            }

        y_true = np.array([1 if p['actual_outcome'] else 0 for p in recent])
        y_scores = np.array([p['probability'] for p in recent])

        thresholds = np.linspace(0, 1, 101)
        precision_list = []
        recall_list = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        # Calculate average precision (area under PR curve)
        ap = 0.0
        for i in range(len(recall_list) - 1):
            if recall_list[i+1] != recall_list[i]:
                ap += abs(recall_list[i+1] - recall_list[i]) * (precision_list[i] + precision_list[i+1]) / 2

        return {
            'precision': [float(x) for x in precision_list],
            'recall': [float(x) for x in recall_list],
            'thresholds': [float(x) for x in thresholds],
            'average_precision': float(ap),
            'insufficient_data': False,
            'sample_count': len(recent)
        }

    def get_calibration_curve(self, hours: int = 50, n_bins: int = 10) -> dict[str, Any]:
        """
        Calculate probability calibration curve.

        Args:
            hours: Number of hours to look back
            n_bins: Number of bins for grouping probabilities

        Returns:
            Dictionary with predicted probabilities, actual frequencies, and calibration score
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if len(recent) < 10:
            return {
                'predicted_probs': [],
                'actual_freq': [],
                'bin_counts': [],
                'calibration_score': 0.0,
                'insufficient_data': True,
                'sample_count': len(recent)
            }

        y_true = np.array([1 if p['actual_outcome'] else 0 for p in recent])
        y_scores = np.array([p['probability'] for p in recent])

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        actual_freqs = []
        bin_counts = []

        for i in range(n_bins):
            bin_mask = (y_scores >= bins[i]) & (y_scores < bins[i+1])

            if i == n_bins - 1:  # Last bin includes upper bound
                bin_mask = (y_scores >= bins[i]) & (y_scores <= bins[i+1])

            if np.sum(bin_mask) > 0:
                bin_center = (bins[i] + bins[i+1]) / 2
                actual_freq = np.mean(y_true[bin_mask])

                bin_centers.append(float(bin_center))
                actual_freqs.append(float(actual_freq))
                bin_counts.append(int(np.sum(bin_mask)))

        # Calculate calibration error (Expected Calibration Error)
        calibration_error = 0.0
        total_samples = len(recent)

        for pred, actual, count in zip(bin_centers, actual_freqs, bin_counts):
            calibration_error += (count / total_samples) * abs(pred - actual)

        calibration_score = 1.0 - calibration_error

        return {
            'predicted_probs': bin_centers,
            'actual_freq': actual_freqs,
            'bin_counts': bin_counts,
            'calibration_score': float(max(0.0, calibration_score)),
            'insufficient_data': False,
            'sample_count': len(recent)
        }

    def get_performance_over_time(self, hours: int = 50, window_hours: int = 5) -> list[dict[str, Any]]:
        """
        Get performance metrics over time using rolling window.

        Args:
            hours: Total hours to look back
            window_hours: Size of rolling window

        Returns:
            List of performance snapshots over time
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if len(recent) < 10:
            return []

        # Sort by timestamp
        recent_sorted = sorted(recent, key=lambda x: x['timestamp'])

        # Create time windows
        performance_timeline = []
        window_delta = timedelta(hours=window_hours)

        start_time = recent_sorted[0]['timestamp']
        end_time = recent_sorted[-1]['timestamp']
        current_time = start_time + window_delta

        while current_time <= end_time:
            window_start = current_time - window_delta

            window_preds = [
                p for p in recent_sorted
                if window_start <= p['timestamp'] < current_time
            ]

            if len(window_preds) >= 5:  # Minimum 5 predictions for meaningful stats
                y_true = [1 if p['actual_outcome'] else 0 for p in window_preds]
                y_pred = [1 if p['probability'] >= 0.5 else 0 for p in window_preds]

                accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

                performance_timeline.append({
                    'timestamp': current_time.isoformat(),
                    'accuracy': float(accuracy),
                    'sample_count': len(window_preds)
                })

            current_time += timedelta(hours=1)  # Move window by 1 hour

        return performance_timeline

    def get_backtest_results(self, hours: int = 50) -> dict[str, Any]:
        """
        Generate realistic backtest results from prediction history with actual price data.

        Uses 1-hour holding constraint:
        - Entry: Price at prediction timestamp
        - Exit: Price exactly 1 hour later
        - No intermediate trading allowed
        - Returns calculated from actual price movements

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with equity curve, metrics, and trade distribution
        """
        from sqlalchemy import select
        from src.utils.database import get_db, MinuteBar

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter to recent predictions with known outcomes
        recent = [
            pred for pred in self.prediction_history
            if pred['timestamp'] >= cutoff_time and pred['actual_outcome'] is not None
        ]

        if not recent:
            return {
                'equity_curve': [],
                'metrics': {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'avg_return': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'calmar_ratio': 0.0,
                    'profit_factor': 0.0
                },
                'trade_distribution': [],
                'recent_trades': [],
                'skipped_trades': 0
            }

        # Sort by timestamp
        recent = sorted(recent, key=lambda x: x['timestamp'])

        commission = settings.COMMISSION_PERCENT * 2  # Round-trip commission

        equity_curve = [{'timestamp': recent[0]['timestamp'].isoformat(), 'equity': 100.0}]
        current_equity = 100.0
        trades = []
        returns = []
        skipped_count = 0
        last_exit_time = None

        with get_db() as db:
            for pred in recent:
                pred_time = pred['timestamp']

                # Enforce 1-hour holding constraint: skip if still in previous position
                if last_exit_time and pred_time < last_exit_time:
                    skipped_count += 1
                    continue

                # Calculate exit time (1 hour after prediction)
                exit_time = pred_time + timedelta(hours=1)

                # Query entry price (at prediction time)
                entry_bar_stmt = (
                    select(MinuteBar)
                    .where(MinuteBar.symbol == self.ticker)
                    .where(MinuteBar.timestamp >= pred_time)
                    .order_by(MinuteBar.timestamp)
                    .limit(1)
                )
                entry_bar = db.execute(entry_bar_stmt).scalar_one_or_none()

                # Query exit price (1 hour later)
                exit_bar_stmt = (
                    select(MinuteBar)
                    .where(MinuteBar.symbol == self.ticker)
                    .where(MinuteBar.timestamp >= exit_time)
                    .order_by(MinuteBar.timestamp)
                    .limit(1)
                )
                exit_bar = db.execute(exit_bar_stmt).scalar_one_or_none()

                # Check if we have valid MinuteBar data
                use_fallback = False
                if not entry_bar or not exit_bar:
                    use_fallback = True
                elif entry_bar.timestamp >= exit_bar.timestamp:
                    use_fallback = True
                elif (exit_bar.timestamp - entry_bar.timestamp).total_seconds() / 60 < 30:
                    use_fallback = True

                # Determine position direction based on prediction
                predicted_up = pred['probability'] >= settings.PROBABILITY_THRESHOLD

                if use_fallback:
                    # Fallback: use actual_outcome from prediction history
                    # This ensures consistency with Overview tab's win_rate
                    actual_outcome = pred.get('actual_outcome')
                    if actual_outcome is None:
                        skipped_count += 1
                        continue

                    # Estimate return based on outcome (target is 5% move)
                    # If prediction was correct, assume positive return
                    # If prediction was wrong, assume negative return
                    estimated_return = 2.5 if actual_outcome else -2.5  # Conservative estimate
                    gross_return_pct = estimated_return
                    entry_price = None
                    exit_price = None
                else:
                    # Calculate actual return from MinuteBar data
                    entry_price = entry_bar.close
                    exit_price = exit_bar.close

                    # Calculate gross return
                    if predicted_up:
                        # Long position: profit when price goes up
                        gross_return_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        # Short position: profit when price goes down
                        gross_return_pct = ((entry_price - exit_price) / entry_price) * 100

                # Apply commission
                net_return_pct = gross_return_pct - commission
                returns.append(net_return_pct)

                # Update equity
                current_equity *= (1 + net_return_pct / 100)
                equity_curve.append({
                    'timestamp': exit_time.isoformat(),
                    'equity': round(current_equity, 2)
                })

                # Record trade
                # Classify: Win if positive, Loss if negative, Neutral if zero
                if net_return_pct > 0:
                    trade_result = 'win'
                elif net_return_pct < 0:
                    trade_result = 'loss'
                else:
                    trade_result = 'neutral'

                trades.append({
                    'timestamp': pred_time.isoformat(),
                    'entry_time': pred_time.isoformat(),
                    'exit_time': exit_time.isoformat(),
                    'entry_price': round(entry_price, 2) if entry_price else None,
                    'exit_price': round(exit_price, 2) if exit_price else None,
                    'return_pct': round(net_return_pct, 2),
                    'is_win': net_return_pct > 0,
                    'is_neutral': net_return_pct == 0,
                    'result': trade_result,
                    'probability': pred['probability'],
                    'direction': 'LONG' if predicted_up else 'SHORT',
                    'is_estimated': use_fallback  # Flag to indicate fallback was used
                })

                # Update last exit time for holding constraint
                last_exit_time = exit_time

        # Calculate performance metrics
        if not returns:
            return {
                'equity_curve': equity_curve,
                'metrics': {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_return': 0.0,
                    'avg_return': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'calmar_ratio': 0.0,
                    'profit_factor': 0.0
                },
                'trade_distribution': [],
                'recent_trades': [],
                'skipped_trades': skipped_count
            }

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        neutrals = [r for r in returns if r == 0]

        total_trades = len(returns)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        total_return = sum(returns)
        avg_return = np.mean(returns) if returns else 0.0

        # Sharpe ratio (annualized for hourly returns)
        if len(returns) > 1:
            returns_std = np.std(returns, ddof=1)
            sharpe = (avg_return / returns_std) * np.sqrt(252 * 6.5) if returns_std > 0 else 0.0
        else:
            sharpe = 0.0

        # Sortino ratio (annualized, downside deviation only)
        if losses:
            downside_std = np.std(losses, ddof=1) if len(losses) > 1 else abs(losses[0])
            sortino = (avg_return / downside_std) * np.sqrt(252 * 6.5) if downside_std > 0 else 0.0
        else:
            sortino = float('inf') if avg_return > 0 else 0.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

        # Calmar ratio (annualized return / max drawdown)
        annualized_return = total_return * (252 * 6.5 / len(returns)) if len(returns) > 0 else 0.0
        calmar = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Trade distribution (histogram data)
        if returns:
            min_return = min(returns)
            max_return = max(returns)
            n_bins = min(10, len(returns))

            if min_return < max_return:
                bins = np.linspace(min_return, max_return, n_bins + 1)
                hist, _ = np.histogram(returns, bins=bins)

                trade_distribution = [
                    {
                        'range': f"{bins[i]:.1f} to {bins[i+1]:.1f}%",
                        'count': int(hist[i]),
                        'min': round(bins[i], 1),
                        'max': round(bins[i+1], 1)
                    }
                    for i in range(len(hist))
                ]
            else:
                trade_distribution = [{
                    'range': f"{min_return:.1f}%",
                    'count': len(returns),
                    'min': round(min_return, 1),
                    'max': round(min_return, 1)
                }]
        else:
            trade_distribution = []

        # Recent trades (last 10)
        recent_trades = trades[-10:][::-1]

        return {
            'equity_curve': equity_curve,
            'metrics': {
                'total_trades': total_trades,
                'win_rate': round(win_rate, 4),
                'total_return': round(total_return, 2),
                'avg_return': round(avg_return, 2),
                'avg_win': round(np.mean(wins), 2) if wins else 0.0,
                'avg_loss': round(np.mean(losses), 2) if losses else 0.0,
                'best_trade': round(max(returns), 2) if returns else 0.0,
                'worst_trade': round(min(returns), 2) if returns else 0.0,
                'sharpe_ratio': round(sharpe, 2),
                'sortino_ratio': round(sortino, 2) if sortino != float('inf') else 999.0,
                'max_drawdown': round(max_drawdown, 2),
                'calmar_ratio': round(calmar, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.0
            },
            'trade_distribution': trade_distribution,
            'recent_trades': recent_trades,
            'skipped_trades': skipped_count
        }

    def __repr__(self) -> str:
        """String representation of model."""
        return (f"{self.__class__.__name__}(ticker={self.ticker}, "
                f"target={self.target}, trained={self.is_trained})")
