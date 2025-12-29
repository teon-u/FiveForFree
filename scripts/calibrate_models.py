"""
Model Calibration Script

Applies probability calibration to existing trained models:
1. Loads validation predictions from each model
2. Fits calibrators (Platt or Isotonic)
3. Evaluates improvement with Brier Score
4. Saves calibrated model wrappers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

from loguru import logger
from config.settings import settings
from src.utils.calibrator import (
    ProbabilityCalibrator,
    compare_calibration_methods,
    print_method_comparison,
    CalibrationMetrics
)


# Paths
DATA_DIR = Path(settings.DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
CALIBRATORS_DIR = DATA_DIR / "calibrators"


class ModelCalibrationManager:
    """Manages calibration for all models."""

    def __init__(self):
        self.calibrators_dir = CALIBRATORS_DIR
        self.calibrators_dir.mkdir(parents=True, exist_ok=True)
        self._calibrators: Dict[str, ProbabilityCalibrator] = {}

    def calibrate_model(
        self,
        ticker: str,
        target: str,
        model_type: str,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        method: str = 'auto'
    ) -> Tuple[ProbabilityCalibrator, CalibrationMetrics]:
        """
        Calibrate a single model.

        Args:
            ticker: Stock ticker
            target: Prediction target (up/down)
            model_type: Model type (xgboost, lightgbm, etc.)
            val_probs: Validation predictions
            val_labels: Validation labels
            method: Calibration method

        Returns:
            Tuple of (calibrator, metrics)
        """
        key = f"{ticker}_{target}_{model_type}"

        # Create and fit calibrator
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(val_probs, val_labels)

        # Save calibrator
        calibrator_path = self.calibrators_dir / f"{key}_calibrator.pkl"
        calibrator.save(calibrator_path)

        # Store in memory
        self._calibrators[key] = calibrator

        logger.info(f"Calibrated {key}: Brier improved from "
                   f"{calibrator._before_metrics.brier_score:.4f} to "
                   f"{calibrator._after_metrics.brier_score:.4f}")

        return calibrator, calibrator._after_metrics

    def get_calibrator(
        self,
        ticker: str,
        target: str,
        model_type: str
    ) -> Optional[ProbabilityCalibrator]:
        """Get calibrator for a model."""
        key = f"{ticker}_{target}_{model_type}"

        # Check memory cache
        if key in self._calibrators:
            return self._calibrators[key]

        # Try loading from disk
        calibrator_path = self.calibrators_dir / f"{key}_calibrator.pkl"
        if calibrator_path.exists():
            calibrator = ProbabilityCalibrator.load(calibrator_path)
            self._calibrators[key] = calibrator
            return calibrator

        return None

    def calibrate_all_models(
        self,
        method: str = 'auto'
    ) -> Dict[str, Dict]:
        """
        Calibrate all trained models using their validation data.

        Returns dictionary with calibration results per model.
        """
        results = {}

        from src.models.model_manager import ModelManager
        from src.utils.database import Database

        mm = ModelManager()
        db = Database()

        tickers = mm.get_tickers()
        logger.info(f"Calibrating models for {len(tickers)} tickers")

        for ticker in tickers:
            ticker_results = {}

            for target in settings.PREDICTION_TARGETS:
                target_results = {}

                try:
                    models = mm.get_all_models(ticker, target)

                    for model_type, model in models.items():
                        if not model.is_trained:
                            continue

                        # Get validation data
                        val_data = self._get_validation_data(
                            ticker, target, model_type, db
                        )

                        if val_data is None:
                            logger.warning(f"No validation data for {ticker}/{target}/{model_type}")
                            continue

                        val_probs, val_labels = val_data

                        if len(val_probs) < 50:
                            logger.warning(f"Insufficient validation data ({len(val_probs)} samples)")
                            continue

                        # Calibrate
                        calibrator, metrics = self.calibrate_model(
                            ticker, target, model_type,
                            val_probs, val_labels,
                            method=method
                        )

                        target_results[model_type] = {
                            'brier_before': calibrator._before_metrics.brier_score,
                            'brier_after': metrics.brier_score,
                            'ece_before': calibrator._before_metrics.ece,
                            'ece_after': metrics.ece,
                            'improvement': (
                                (calibrator._before_metrics.brier_score - metrics.brier_score)
                                / calibrator._before_metrics.brier_score * 100
                            )
                        }

                except Exception as e:
                    logger.error(f"Failed to calibrate {ticker}/{target}: {e}")

                if target_results:
                    ticker_results[target] = target_results

            if ticker_results:
                results[ticker] = ticker_results

        return results

    def _get_validation_data(
        self,
        ticker: str,
        target: str,
        model_type: str,
        db
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get validation predictions and labels for a model.

        This uses the model's recorded predictions with outcomes.
        """
        try:
            from src.models.model_manager import ModelManager

            mm = ModelManager()
            model = mm.get_or_create_model(ticker, model_type, target)

            if not model.is_trained:
                return None

            # Get prediction history with outcomes
            history = model.get_prediction_history()

            if not history or len(history) < 50:
                return None

            # Extract probabilities and outcomes
            probs = []
            labels = []

            for record in history:
                if 'probability' in record and 'outcome' in record:
                    if record['outcome'] is not None:
                        probs.append(record['probability'])
                        labels.append(1 if record['outcome'] else 0)

            if len(probs) < 50:
                return None

            return np.array(probs), np.array(labels)

        except Exception as e:
            logger.debug(f"Could not get validation data: {e}")
            return None

    def generate_report(self, results: Dict) -> str:
        """Generate calibration report."""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("  MODEL CALIBRATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")

        total_models = 0
        improved_models = 0
        total_improvement = 0.0

        for ticker, ticker_results in results.items():
            report_lines.append(f"\n{ticker}")
            report_lines.append("-" * 40)

            for target, target_results in ticker_results.items():
                for model_type, metrics in target_results.items():
                    total_models += 1
                    improvement = metrics['improvement']

                    if improvement > 0:
                        improved_models += 1
                        total_improvement += improvement

                    status = "+" if improvement > 0 else "-"
                    report_lines.append(
                        f"  {target:5s} {model_type:12s}: "
                        f"Brier {metrics['brier_before']:.4f} -> {metrics['brier_after']:.4f} "
                        f"({status}{abs(improvement):.1f}%)"
                    )

        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append(f"Total models calibrated: {total_models}")
        report_lines.append(f"Models improved: {improved_models} ({improved_models/total_models*100:.1f}%)" if total_models > 0 else "Models improved: 0")
        report_lines.append(f"Average improvement: {total_improvement/improved_models:.1f}%" if improved_models > 0 else "Average improvement: N/A")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)


class CalibratedModelWrapper:
    """Wrapper that applies calibration to model predictions."""

    def __init__(self, base_model, calibrator: ProbabilityCalibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        """Get calibrated probability predictions."""
        raw_probs = self.base_model.predict_proba(X)
        return self.calibrator.transform(raw_probs)

    def predict(self, X) -> np.ndarray:
        """Get binary predictions using calibrated probabilities."""
        probs = self.predict_proba(X)
        threshold = settings.PROBABILITY_THRESHOLD
        return (probs >= threshold).astype(int)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Calibration")
    parser.add_argument("--method", choices=['platt', 'isotonic', 'auto'],
                        default='auto', help="Calibration method")
    parser.add_argument("--ticker", type=str,
                        help="Calibrate specific ticker only")
    parser.add_argument("--compare", action="store_true",
                        help="Compare calibration methods")
    parser.add_argument("--report", type=str,
                        help="Save report to file")

    args = parser.parse_args()

    manager = ModelCalibrationManager()

    if args.compare:
        print("\nComparing calibration methods...")
        # Use sample data for comparison
        np.random.seed(42)
        n = 1000
        labels = np.random.binomial(1, 0.3, n)
        probs = 0.5 + (labels - 0.5) * 0.5 + np.random.normal(0, 0.1, n)
        probs = np.clip(probs, 0.01, 0.99)

        results = compare_calibration_methods(
            probs[:700], labels[:700],
            probs[700:], labels[700:]
        )
        print_method_comparison(results)
        return

    print("\nCalibrating models...")
    results = manager.calibrate_all_models(method=args.method)

    if not results:
        print("No models to calibrate (no validation data available)")
        return

    report = manager.generate_report(results)
    print(report)

    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.report}")

    # Save results as JSON
    results_path = CALIBRATORS_DIR / "calibration_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
