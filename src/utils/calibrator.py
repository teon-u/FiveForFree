"""
Probability Calibration Module

Provides calibration methods to improve probability estimates:
- Platt Scaling (Sigmoid): Fast, works well for small datasets
- Isotonic Regression: Non-parametric, better for large datasets

Usage:
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.fit(val_probs, val_labels)
    calibrated_probs = calibrator.transform(test_probs)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit  # Sigmoid function
from loguru import logger

from config.settings import settings


@dataclass
class CalibrationMetrics:
    """Calibration evaluation metrics."""
    brier_score: float  # Lower is better (0-1)
    log_loss_val: float  # Lower is better
    ece: float  # Expected Calibration Error (lower is better)
    mce: float  # Maximum Calibration Error (lower is better)
    reliability_diagram: Dict  # For visualization


class PlattScaler:
    """
    Platt Scaling using sigmoid function.

    Fits a logistic regression on the probability outputs
    to learn optimal sigmoid parameters A, B:
        P_calibrated = 1 / (1 + exp(A * P_original + B))
    """

    def __init__(self):
        self.A = 0.0
        self.B = 0.0
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'PlattScaler':
        """
        Fit Platt scaling parameters.

        Args:
            probs: Original predicted probabilities
            labels: True binary labels
        """
        # Use logistic regression to find optimal parameters
        # Reshape for sklearn
        X = probs.reshape(-1, 1)
        y = labels.ravel()

        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X, y)

        # Extract parameters
        self.A = lr.coef_[0][0]
        self.B = lr.intercept_[0]
        self._fitted = True

        logger.info(f"Platt Scaling fitted: A={self.A:.4f}, B={self.B:.4f}")
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities."""
        if not self._fitted:
            raise ValueError("PlattScaler not fitted. Call fit() first.")

        # Apply sigmoid transformation
        calibrated = expit(self.A * probs + self.B)
        return calibrated

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probs, labels)
        return self.transform(probs)


class IsotonicCalibrator:
    """
    Isotonic Regression calibrator.

    Non-parametric, monotonic calibration that learns
    a step-wise mapping from original to calibrated probabilities.
    """

    def __init__(self, out_of_bounds: str = 'clip'):
        self.ir = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic regression.

        Args:
            probs: Original predicted probabilities
            labels: True binary labels
        """
        self.ir.fit(probs.ravel(), labels.ravel())
        self._fitted = True
        logger.info("Isotonic calibrator fitted")
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to probabilities."""
        if not self._fitted:
            raise ValueError("IsotonicCalibrator not fitted. Call fit() first.")

        return self.ir.predict(probs.ravel())

    def fit_transform(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probs, labels)
        return self.transform(probs)


class ProbabilityCalibrator:
    """
    Unified probability calibrator with method selection.

    Supports:
    - 'platt': Platt Scaling (sigmoid-based)
    - 'isotonic': Isotonic Regression
    - 'auto': Automatically select based on data size
    """

    def __init__(
        self,
        method: Literal['platt', 'isotonic', 'auto'] = None
    ):
        self.method = method or settings.CALIBRATION_METHOD
        self._calibrator = None
        self._fitted = False
        self._before_metrics = None
        self._after_metrics = None

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        method: str = None
    ) -> 'ProbabilityCalibrator':
        """
        Fit the calibrator.

        Args:
            probs: Original predicted probabilities
            labels: True binary labels
            method: Override method selection
        """
        method = method or self.method

        # Auto-select based on data size
        if method == 'auto':
            # Isotonic needs more data for stable estimates
            method = 'isotonic' if len(probs) > 1000 else 'platt'
            logger.info(f"Auto-selected calibration method: {method}")

        # Store before metrics
        self._before_metrics = self._calculate_metrics(probs, labels)

        # Create and fit calibrator
        if method == 'platt':
            self._calibrator = PlattScaler()
        elif method == 'isotonic':
            self._calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self._calibrator.fit(probs, labels)
        self._fitted = True
        self.method = method

        # Calculate after metrics (in-sample, for reference)
        calibrated = self._calibrator.transform(probs)
        self._after_metrics = self._calculate_metrics(calibrated, labels)

        logger.info(
            f"Calibration complete ({method}): "
            f"Brier {self._before_metrics.brier_score:.4f} -> "
            f"{self._after_metrics.brier_score:.4f}"
        )

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self._calibrator.transform(probs)

    def fit_transform(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(probs, labels)
        return self.transform(probs)

    def _calculate_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """Calculate calibration metrics."""
        probs = probs.ravel()
        labels = labels.ravel()

        # Brier score
        brier = brier_score_loss(labels, probs)

        # Log loss
        try:
            ll = log_loss(labels, probs, labels=[0, 1])
        except Exception:
            ll = float('inf')

        # Calibration curve for ECE/MCE
        prob_true, prob_pred = calibration_curve(
            labels, probs, n_bins=n_bins, strategy='uniform'
        )

        # Expected Calibration Error (weighted by bin count)
        bin_counts = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
        total = len(probs)
        ece = 0.0
        mce = 0.0

        for i, (true_prob, pred_prob) in enumerate(zip(prob_true, prob_pred)):
            if i < len(bin_counts):
                weight = bin_counts[i] / total
                diff = abs(true_prob - pred_prob)
                ece += weight * diff
                mce = max(mce, diff)

        return CalibrationMetrics(
            brier_score=brier,
            log_loss_val=ll,
            ece=ece,
            mce=mce,
            reliability_diagram={
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist(),
                'bin_counts': bin_counts.tolist()
            }
        )

    def evaluate(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, CalibrationMetrics]:
        """
        Evaluate calibration on test data.

        Returns metrics before and after calibration.
        """
        if not self._fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        before = self._calculate_metrics(test_probs, test_labels)
        calibrated = self.transform(test_probs)
        after = self._calculate_metrics(calibrated, test_labels)

        return {
            'before': before,
            'after': after,
            'improvement': {
                'brier_reduction': before.brier_score - after.brier_score,
                'ece_reduction': before.ece - after.ece,
                'brier_pct_improvement': (
                    (before.brier_score - after.brier_score) / before.brier_score * 100
                    if before.brier_score > 0 else 0
                )
            }
        }

    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        if not self._fitted:
            raise ValueError("Cannot save unfitted calibrator")

        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self._calibrator,
                'before_metrics': self._before_metrics,
                'after_metrics': self._after_metrics
            }, f)

        logger.info(f"Calibrator saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'ProbabilityCalibrator':
        """Load calibrator from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls(method=data['method'])
        calibrator._calibrator = data['calibrator']
        calibrator._fitted = True
        calibrator._before_metrics = data['before_metrics']
        calibrator._after_metrics = data['after_metrics']

        logger.info(f"Calibrator loaded from {path}")
        return calibrator

    def print_comparison(self) -> None:
        """Print before/after comparison."""
        if not self._fitted:
            print("Calibrator not fitted")
            return

        print("\n" + "=" * 60)
        print("  CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Method: {self.method.upper()}")
        print("-" * 60)
        print(f"{'Metric':<20} {'Before':>15} {'After':>15} {'Improvement':>12}")
        print("-" * 60)

        b = self._before_metrics
        a = self._after_metrics

        metrics = [
            ('Brier Score', b.brier_score, a.brier_score),
            ('Log Loss', b.log_loss_val, a.log_loss_val),
            ('ECE', b.ece, a.ece),
            ('MCE', b.mce, a.mce),
        ]

        for name, before, after in metrics:
            if before == float('inf') or after == float('inf'):
                continue
            improvement = before - after
            pct = (improvement / before * 100) if before > 0 else 0
            print(f"{name:<20} {before:>15.4f} {after:>15.4f} {pct:>10.1f}%")

        print("=" * 60)


def compare_calibration_methods(
    probs: np.ndarray,
    labels: np.ndarray,
    test_probs: np.ndarray = None,
    test_labels: np.ndarray = None
) -> Dict[str, Dict]:
    """
    Compare Platt and Isotonic calibration methods.

    Args:
        probs: Training probabilities
        labels: Training labels
        test_probs: Optional test probabilities
        test_labels: Optional test labels

    Returns:
        Comparison results for both methods
    """
    results = {}

    for method in ['platt', 'isotonic']:
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(probs, labels)

        if test_probs is not None and test_labels is not None:
            eval_result = calibrator.evaluate(test_probs, test_labels)
            results[method] = {
                'train_metrics': {
                    'brier_before': calibrator._before_metrics.brier_score,
                    'brier_after': calibrator._after_metrics.brier_score,
                },
                'test_metrics': {
                    'brier_before': eval_result['before'].brier_score,
                    'brier_after': eval_result['after'].brier_score,
                    'ece_before': eval_result['before'].ece,
                    'ece_after': eval_result['after'].ece,
                },
                'improvement': eval_result['improvement']
            }
        else:
            results[method] = {
                'train_metrics': {
                    'brier_before': calibrator._before_metrics.brier_score,
                    'brier_after': calibrator._after_metrics.brier_score,
                    'ece_before': calibrator._before_metrics.ece,
                    'ece_after': calibrator._after_metrics.ece,
                }
            }

    return results


def print_method_comparison(results: Dict[str, Dict]) -> None:
    """Print comparison between calibration methods."""
    print("\n" + "=" * 70)
    print("  CALIBRATION METHOD COMPARISON")
    print("=" * 70)
    print(f"{'Method':<12} | {'Brier (before)':>14} | {'Brier (after)':>14} | {'Improvement':>12}")
    print("-" * 70)

    for method, data in results.items():
        metrics = data.get('test_metrics', data.get('train_metrics', {}))
        before = metrics.get('brier_before', 0)
        after = metrics.get('brier_after', 0)
        imp = (before - after) / before * 100 if before > 0 else 0

        print(f"{method.upper():<12} | {before:>14.4f} | {after:>14.4f} | {imp:>10.1f}%")

    print("=" * 70)

    # Recommendation
    platt_imp = results.get('platt', {}).get('improvement', {}).get('brier_pct_improvement', 0)
    iso_imp = results.get('isotonic', {}).get('improvement', {}).get('brier_pct_improvement', 0)

    if iso_imp > platt_imp:
        print("Recommendation: ISOTONIC (better calibration)")
    else:
        print("Recommendation: PLATT (more stable, sufficient calibration)")


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    # Generate synthetic uncalibrated probabilities
    n_samples = 2000
    true_labels = np.random.binomial(1, 0.3, n_samples)

    # Simulate overconfident predictions
    noise = np.random.normal(0, 0.15, n_samples)
    raw_probs = 0.5 + (true_labels - 0.5) * 0.6 + noise
    raw_probs = np.clip(raw_probs, 0.01, 0.99)

    # Split into train/test
    split = int(n_samples * 0.7)
    train_probs, test_probs = raw_probs[:split], raw_probs[split:]
    train_labels, test_labels = true_labels[:split], true_labels[split:]

    # Compare methods
    print("\nComparing calibration methods...")
    results = compare_calibration_methods(
        train_probs, train_labels,
        test_probs, test_labels
    )
    print_method_comparison(results)

    # Demo single calibrator
    print("\n\nDemo: Isotonic Calibrator")
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.fit(train_probs, train_labels)
    calibrator.print_comparison()
