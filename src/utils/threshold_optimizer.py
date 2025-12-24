"""Threshold optimization utilities for signal rate improvement."""

import numpy as np
from sklearn.metrics import precision_recall_curve
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config.settings import settings


def calculate_class_weight(y_train: np.ndarray, max_weight: float = None) -> float:
    """
    Calculate class weight for imbalanced dataset.

    Args:
        y_train: Training labels array
        max_weight: Maximum weight (default from settings.MAX_CLASS_WEIGHT)

    Returns:
        scale_pos_weight value
    """
    if max_weight is None:
        max_weight = settings.MAX_CLASS_WEIGHT

    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()

    if n_positive == 0:
        logger.warning("No positive samples found, returning max_weight")
        return max_weight

    weight = n_negative / n_positive

    # Apply cap to prevent overfitting
    capped_weight = min(weight, max_weight)

    logger.info(
        f"Class weight calculated: {capped_weight:.2f} "
        f"(raw={weight:.2f}, pos={n_positive}, neg={n_negative})"
    )

    return capped_weight


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_precision: float = None,
    min_signal_rate: float = None
) -> float:
    """
    Find optimal threshold satisfying minimum precision and signal rate.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        min_precision: Minimum required precision (default from settings)
        min_signal_rate: Minimum required signal rate (default from settings)

    Returns:
        Optimal threshold (default: 0.5)
    """
    if min_precision is None:
        min_precision = settings.MIN_PRECISION
    if min_signal_rate is None:
        min_signal_rate = settings.MIN_SIGNAL_RATE

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    best_threshold = settings.SIGNAL_THRESHOLD  # Default
    best_f1 = 0.0

    for i, thresh in enumerate(thresholds):
        # Calculate signal rate
        signal_rate = (y_pred_proba >= thresh).mean()

        # Check if conditions are met
        if precisions[i] >= min_precision and signal_rate >= min_signal_rate:
            # Calculate F1 Score
            if precisions[i] + recalls[i] > 0:
                f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

    logger.info(
        f"Optimal threshold found: {best_threshold:.3f} "
        f"(F1={best_f1:.3f}, min_prec={min_precision}, min_sr={min_signal_rate})"
    )

    return best_threshold


def evaluate_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: List[float] = None
) -> Dict[float, Dict]:
    """
    Evaluate metrics for multiple thresholds.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary with metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results = {}

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        signal_rate = y_pred.mean()
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Check if meets minimum requirements
        meets_requirements = (
            precision >= settings.MIN_PRECISION and
            signal_rate >= settings.MIN_SIGNAL_RATE
        )

        results[thresh] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'signal_rate': signal_rate,
            'signals_generated': int(y_pred.sum()),
            'total_samples': len(y_pred),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'meets_requirements': meets_requirements
        }

    return results


def print_threshold_comparison(results: Dict[float, Dict]) -> None:
    """Print threshold comparison table."""
    print("\n" + "=" * 80)
    print("Threshold Comparison")
    print("=" * 80)
    print(f"{'Thresh':>7} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | "
          f"{'SigRate':>7} | {'Signals':>8} | {'Status':>6}")
    print("-" * 80)

    for thresh, metrics in sorted(results.items()):
        status = "OK" if metrics['meets_requirements'] else "FAIL"
        print(
            f"{thresh:>7.2f} | "
            f"{metrics['precision']*100:>5.1f}% | "
            f"{metrics['recall']*100:>5.1f}% | "
            f"{metrics['f1']*100:>5.1f}% | "
            f"{metrics['signal_rate']*100:>6.1f}% | "
            f"{metrics['signals_generated']:>8} | "
            f"{status:>6}"
        )

    print("=" * 80)
    print(f"Requirements: Precision >= {settings.MIN_PRECISION*100}%, "
          f"Signal Rate >= {settings.MIN_SIGNAL_RATE*100}%")
