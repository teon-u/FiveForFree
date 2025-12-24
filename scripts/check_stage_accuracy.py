#!/usr/bin/env python3
"""
Check Stage 1 (Volatility) and Stage 2 (Direction) model accuracy.

Usage:
    python scripts/check_stage_accuracy.py
    python scripts/check_stage_accuracy.py --ticker AAPL
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=True)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import select

from src.models.model_manager import ModelManager
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator
from config.settings import settings


def load_ticker_data(ticker: str, days: int = 60):
    """Load historical data for a ticker."""
    cutoff_date = datetime.now() - timedelta(days=days)

    with get_db() as db:
        stmt = (
            select(DBMinuteBar)
            .where(DBMinuteBar.symbol == ticker)
            .where(DBMinuteBar.timestamp >= cutoff_date)
            .order_by(DBMinuteBar.timestamp)
        )
        bars = db.execute(stmt).scalars().all()

        if not bars:
            return None

        df = pd.DataFrame([{
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
        } for bar in bars])

        return df


def evaluate_models(ticker: str, model_manager: ModelManager):
    """Evaluate Stage 1 and Stage 2 models for a ticker."""
    print(f"\n{'='*60}")
    print(f"Evaluating models for {ticker}")
    print(f"{'='*60}")

    # Load data
    df = load_ticker_data(ticker)
    if df is None or len(df) < 500:
        print(f"  Insufficient data for {ticker}")
        return None

    # Prepare features and labels
    feature_engineer = FeatureEngineer()
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    features_df = feature_engineer.compute_features(df)
    feature_names = feature_engineer.get_feature_names()

    # Generate labels
    labels_vol = []
    labels_dir = []

    for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
        entry_time = df.iloc[idx]["timestamp"]
        entry_price = df.iloc[idx]["close"]
        labels = label_generator.generate_labels(df, entry_time, entry_price)
        labels_vol.append(labels["label_volatility"])
        labels_dir.append(labels["label_direction"])

    X = features_df[feature_names].values[:len(labels_vol)]
    y_vol = np.array(labels_vol)
    y_dir = np.array(labels_dir)

    # Remove NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_vol = y_vol[valid_mask]
    y_dir = y_dir[valid_mask]

    # Use last 20% as test set (same as validation split)
    n_test = int(len(X) * 0.2)
    X_test = X[-n_test:]
    y_vol_test = y_vol[-n_test:]
    y_dir_test = y_dir[-n_test:]

    results = {
        'ticker': ticker,
        'test_samples': len(X_test),
        'volatility': {},
        'direction': {}
    }

    # ===== Stage 1: Volatility Models =====
    print(f"\n[Stage 1] Volatility Models (test samples: {len(X_test)})")
    print(f"  Volatility rate in test: {y_vol_test.mean()*100:.1f}%")
    print("-" * 50)

    for model_type in ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']:
        try:
            _, model = model_manager.get_or_create_model(ticker, model_type, 'volatility')
            if model.is_trained:
                preds = model.predict_proba(X_test)
                preds_binary = (preds > 0.5).astype(int)

                # Handle length mismatch for sequence models
                if len(preds_binary) < len(y_vol_test):
                    y_test_aligned = y_vol_test[-len(preds_binary):]
                else:
                    y_test_aligned = y_vol_test
                    preds_binary = preds_binary[-len(y_test_aligned):]

                accuracy = (preds_binary == y_test_aligned).mean()

                # Calculate precision (when model predicts 1, how often correct)
                pred_positive = preds_binary == 1
                if pred_positive.sum() > 0:
                    precision = (y_test_aligned[pred_positive] == 1).mean()
                else:
                    precision = 0.0

                results['volatility'][model_type] = {
                    'accuracy': accuracy,
                    'precision': precision
                }
                print(f"  {model_type:12s}: Accuracy={accuracy*100:5.1f}%, Precision={precision*100:5.1f}%")
        except Exception as e:
            print(f"  {model_type:12s}: Error - {e}")

    # ===== Stage 2: Direction Models =====
    # Filter to volatile samples only
    volatile_mask = y_vol_test == 1
    X_test_volatile = X_test[volatile_mask]
    y_dir_test_volatile = y_dir_test[volatile_mask]

    print(f"\n[Stage 2] Direction Models (volatile samples only: {len(X_test_volatile)})")
    if len(X_test_volatile) > 0:
        print(f"  Direction UP rate: {y_dir_test_volatile.mean()*100:.1f}%")
    print("-" * 50)

    if len(X_test_volatile) < 10:
        print("  Insufficient volatile samples for evaluation")
        return results

    for model_type in ['xgboost', 'lightgbm', 'lstm', 'transformer', 'ensemble']:
        try:
            _, model = model_manager.get_or_create_model(ticker, model_type, 'direction')
            if model.is_trained:
                preds = model.predict_proba(X_test_volatile)
                preds_binary = (preds > 0.5).astype(int)

                # Handle length mismatch
                if len(preds_binary) < len(y_dir_test_volatile):
                    y_test_aligned = y_dir_test_volatile[-len(preds_binary):]
                else:
                    y_test_aligned = y_dir_test_volatile
                    preds_binary = preds_binary[-len(y_test_aligned):]

                accuracy = (preds_binary == y_test_aligned).mean()

                # Precision for UP prediction
                pred_up = preds_binary == 1
                if pred_up.sum() > 0:
                    precision_up = (y_test_aligned[pred_up] == 1).mean()
                else:
                    precision_up = 0.0

                results['direction'][model_type] = {
                    'accuracy': accuracy,
                    'precision_up': precision_up
                }
                print(f"  {model_type:12s}: Accuracy={accuracy*100:5.1f}%, Precision(UP)={precision_up*100:5.1f}%")
        except Exception as e:
            print(f"  {model_type:12s}: Error - {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Check Stage 1/2 model accuracy")
    parser.add_argument("--ticker", "-t", help="Specific ticker to check")
    parser.add_argument("--top", type=int, default=5, help="Number of tickers to check (default: 5)")
    args = parser.parse_args()

    print("Loading models...")
    model_manager = ModelManager()
    model_manager.load_all_models()

    summary = model_manager.get_summary()
    print(f"Loaded {summary['trained_models']} trained models")

    if args.ticker:
        tickers = [args.ticker]
    else:
        # Get tickers with models
        models_path = Path("data/models")
        if models_path.exists():
            tickers = sorted([d.name for d in models_path.iterdir() if d.is_dir()])[:args.top]
        else:
            print("No models found")
            return

    all_results = []
    for ticker in tickers:
        result = evaluate_models(ticker, model_manager)
        if result:
            all_results.append(result)

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        # Average across tickers
        vol_accs = []
        dir_accs = []

        for r in all_results:
            for model_type, metrics in r['volatility'].items():
                vol_accs.append(metrics['accuracy'])
            for model_type, metrics in r['direction'].items():
                dir_accs.append(metrics['accuracy'])

        if vol_accs:
            print(f"\nStage 1 (Volatility) Average Accuracy: {np.mean(vol_accs)*100:.1f}%")
        if dir_accs:
            print(f"Stage 2 (Direction)  Average Accuracy: {np.mean(dir_accs)*100:.1f}%")


if __name__ == "__main__":
    main()
