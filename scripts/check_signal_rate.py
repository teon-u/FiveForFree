#!/usr/bin/env python
"""Signal Rate 및 핵심 지표 검증 스크립트.

train_all_models.py와 동일한 데이터 로딩 방식 사용.

Usage:
    python scripts/check_signal_rate.py --ticker AAPL
    python scripts/check_signal_rate.py --ticker AAPL --threshold 0.40
    python scripts/check_signal_rate.py --all
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=True)

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select
from sklearn.metrics import precision_score, recall_score, f1_score

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator

warnings.filterwarnings("ignore")


def load_ticker_data(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Load historical minute bar data for a ticker."""
    try:
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
                logger.warning(f"{ticker}: No data found")
                return None

            df = pd.DataFrame([
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                }
                for bar in bars
            ])

            logger.info(f"{ticker}: Loaded {len(df)} bars")
            return df

    except Exception as e:
        logger.error(f"{ticker}: Failed to load data: {e}")
        return None


def prepare_data(ticker: str, df: pd.DataFrame) -> Optional[tuple]:
    """Prepare features and labels for evaluation."""
    try:
        feature_engineer = FeatureEngineer(ticker=ticker)
        label_generator = LabelGenerator(
            target_percent=settings.TARGET_PERCENT,
            prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
        )

        # Compute features
        features_df = feature_engineer.compute_features(df)

        # Generate labels
        labels_up = []
        labels_down = []

        for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
            entry_time = df.iloc[idx]["timestamp"]
            entry_price = df.iloc[idx]["close"]
            labels = label_generator.generate_labels(df, entry_time, entry_price)
            labels_up.append(labels["label_up"])
            labels_down.append(labels["label_down"])

        # Use same feature selection as train_all_models.py
        feature_names = feature_engineer.get_feature_names()
        trim_len = len(labels_up)
        X = features_df[feature_names].values[:trim_len]

        return X, np.array(labels_up), np.array(labels_down)

    except Exception as e:
        logger.error(f"{ticker}: Failed to prepare data: {e}")
        return None


def check_signal_rate_for_ticker(
    ticker: str,
    threshold: float = None,
    model_types: list = None
) -> dict:
    """Check signal rate and key metrics for a ticker."""
    if threshold is None:
        threshold = settings.SIGNAL_THRESHOLD
    if model_types is None:
        model_types = ['xgboost', 'lightgbm']

    model_manager = ModelManager()
    results = {}

    print(f"\n{'=' * 60}")
    print(f"Signal Rate Check: {ticker}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 60}")

    # Load data
    df = load_ticker_data(ticker)
    if df is None:
        print(f"  No data available for {ticker}")
        return results

    # Prepare features and labels
    data = prepare_data(ticker, df)
    if data is None:
        print(f"  Failed to prepare data for {ticker}")
        return results

    X, y_up, y_down = data
    targets = {'up': y_up, 'down': y_down}

    print(f"  Data: {len(X)} samples")
    print(f"  Labels - Up: {y_up.sum()} ({y_up.mean()*100:.1f}%), Down: {y_down.sum()} ({y_down.mean()*100:.1f}%)")

    # Load all models first
    model_manager.load_all_models()

    for model_type in model_types:
        for target_name, y_true in targets.items():
            key = f'{model_type}_{target_name}'

            try:
                # Check if model exists
                if ticker.upper() not in model_manager._models:
                    print(f"\n  [{model_type.upper()} {target_name}] No models for {ticker}")
                    continue

                if model_type not in model_manager._models[ticker.upper()]:
                    print(f"\n  [{model_type.upper()} {target_name}] Model type not found")
                    continue

                if target_name not in model_manager._models[ticker.upper()][model_type]:
                    print(f"\n  [{model_type.upper()} {target_name}] Target not found")
                    continue

                model = model_manager._models[ticker.upper()][model_type][target_name]
                if model is None:
                    print(f"\n  [{model_type.upper()} {target_name}] Model is None")
                    continue

                # Get predictions
                y_proba = model.predict_proba(X)
                y_pred = (y_proba >= threshold).astype(int)

                # Calculate metrics
                signal_rate = y_pred.mean() * 100
                precision = precision_score(y_true, y_pred, zero_division=0) * 100
                recall = recall_score(y_true, y_pred, zero_division=0) * 100
                f1 = f1_score(y_true, y_pred, zero_division=0) * 100

                # Check requirements
                meets_req = (
                    signal_rate >= settings.MIN_SIGNAL_RATE * 100 and
                    precision >= settings.MIN_PRECISION * 100
                )

                results[key] = {
                    'signal_rate': signal_rate,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'signals': int(y_pred.sum()),
                    'total': len(y_pred),
                    'meets_requirements': meets_req
                }

                status = "OK" if meets_req else "FAIL"
                print(f"\n  [{status}] {model_type.upper()} ({target_name})")
                print(f"       Signal Rate: {signal_rate:.1f}% (target: >={settings.MIN_SIGNAL_RATE*100}%)")
                print(f"       Precision:   {precision:.1f}% (target: >={settings.MIN_PRECISION*100}%)")
                print(f"       Recall:      {recall:.1f}%")
                print(f"       F1 Score:    {f1:.1f}%")
                print(f"       Signals:     {y_pred.sum()} / {len(y_pred)}")

                # Show threshold analysis if fails
                if not meets_req:
                    print(f"\n       --- Threshold Analysis ---")
                    for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
                        y_p = (y_proba >= t).astype(int)
                        sr = y_p.mean() * 100
                        pr = precision_score(y_true, y_p, zero_division=0) * 100
                        ok = "OK" if sr >= 10 and pr >= 60 else ""
                        print(f"       t={t:.2f}: SR={sr:.1f}%, Prec={pr:.1f}% {ok}")

            except Exception as e:
                print(f"  [{model_type.upper()} {target_name}] Error - {e}")
                logger.exception(f"Error checking {key}")

    return results


def check_all_tickers(threshold: float = None) -> dict:
    """Check signal rate for all default tickers."""
    all_results = {}

    for ticker in settings.DEFAULT_TICKERS:
        all_results[ticker] = check_signal_rate_for_ticker(ticker, threshold)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    ok_count = 0
    fail_count = 0

    for ticker, ticker_results in all_results.items():
        for model_key, metrics in ticker_results.items():
            if metrics.get('meets_requirements'):
                ok_count += 1
            else:
                fail_count += 1

    print(f"Total models checked: {ok_count + fail_count}")
    print(f"Meeting requirements: {ok_count}")
    print(f"Not meeting requirements: {fail_count}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Check Signal Rate and key metrics for trained models"
    )
    parser.add_argument('--ticker', type=str, help='Ticker symbol to check')
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'Prediction threshold (default: {settings.SIGNAL_THRESHOLD})')
    parser.add_argument('--all', action='store_true', help='Check all default tickers')

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    if args.all:
        check_all_tickers(args.threshold)
    elif args.ticker:
        check_signal_rate_for_ticker(args.ticker, args.threshold)
    else:
        print("Please specify --ticker SYMBOL or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
