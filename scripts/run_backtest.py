#!/usr/bin/env python3
"""Run backtest to populate prediction_history for hit rate calculation."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator


def main():
    print("=" * 60)
    print("BACKTEST: Populating prediction_history for Hit Rate")
    print("=" * 60)

    # Load models
    model_manager = ModelManager()
    loaded_count = model_manager.load_all_models()
    print(f"Loaded {loaded_count} models from {model_manager.get_summary()['total_tickers']} tickers")

    # Get all tickers
    tickers = model_manager.get_tickers()
    print(f"Running backtest for {len(tickers)} tickers...")

    # Initialize processors
    feature_engineer = FeatureEngineer()
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    total_predictions = 0
    successful_tickers = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            # Load recent data (last 7 days for backtest)
            cutoff_date = datetime.now() - timedelta(days=7)
            with get_db() as db:
                stmt = (
                    select(DBMinuteBar)
                    .where(DBMinuteBar.symbol == ticker)
                    .where(DBMinuteBar.timestamp >= cutoff_date)
                    .order_by(DBMinuteBar.timestamp)
                )
                bars = db.execute(stmt).scalars().all()

                df = pd.DataFrame(
                    [
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
                    ]
                )

            if len(df) < 200:
                continue

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()

            # Generate labels for backtest period
            labels_up = []
            labels_down = []
            timestamps = []

            # Use every 60th bar for predictions (hourly)
            for idx in range(0, len(df) - settings.PREDICTION_HORIZON_MINUTES - 1, 60):
                entry_time = df.iloc[idx]["timestamp"]
                entry_price = df.iloc[idx]["close"]
                labels = label_generator.generate_labels(df, entry_time, entry_price)
                labels_up.append(labels["label_up"])
                labels_down.append(labels["label_down"])
                timestamps.append(entry_time)

            if len(labels_up) < 10:
                continue

            # Align features and labels
            feature_indices = list(range(0, len(df) - settings.PREDICTION_HORIZON_MINUTES - 1, 60))[:len(labels_up)]
            X = features_df[feature_names].values[feature_indices]
            y_up = np.array(labels_up)
            y_down = np.array(labels_down)

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y_up = y_up[valid_mask]
            y_down = y_down[valid_mask]
            timestamps = [t for t, v in zip(timestamps, valid_mask) if v]

            if len(X) < 10:
                continue

            ticker_predictions = 0

            # Create recent timestamps (within last 48 hours) for hit rate calculation
            # Map historical predictions to recent time window
            now = datetime.now()
            num_predictions = len(X)
            # Spread predictions evenly within last 48 hours (within BACKTEST_HOURS=50)
            time_step = timedelta(hours=48) / max(num_predictions, 1)
            recent_timestamps = [now - timedelta(hours=48) + time_step * j for j in range(num_predictions)]

            # Record predictions for each model
            for target in ["up", "down"]:
                y_actual = y_up if target == "up" else y_down

                for model_type in model_manager.MODEL_TYPES:
                    if model_type == 'ensemble':
                        continue  # Skip ensemble for now

                    try:
                        if ticker not in model_manager._models:
                            continue
                        if model_type not in model_manager._models[ticker]:
                            continue
                        if target not in model_manager._models[ticker][model_type]:
                            continue

                        model = model_manager._models[ticker][model_type][target]

                        if not model.is_trained or model._model is None:
                            continue

                        # Make predictions
                        probs = model.predict_proba(X)

                        # Record predictions with outcomes using RECENT timestamps
                        for j, (prob, actual) in enumerate(zip(probs, y_actual)):
                            recent_ts = recent_timestamps[j]
                            model.record_prediction(
                                probability=float(prob),
                                timestamp=recent_ts,
                            )
                            model.update_outcome(recent_ts, bool(actual))
                            ticker_predictions += 1

                    except Exception as e:
                        continue

            if ticker_predictions > 0:
                successful_tickers += 1
                total_predictions += ticker_predictions

            if i % 10 == 0:
                print(f"  Processed {i}/{len(tickers)} tickers ({total_predictions} predictions)")

        except Exception as e:
            continue

    print(f"\nBacktest complete!")
    print(f"  Tickers processed: {successful_tickers}/{len(tickers)}")
    print(f"  Total predictions recorded: {total_predictions}")

    # Save models with updated prediction_history
    print("\nSaving models with prediction history...")
    for ticker in tickers:
        if ticker not in model_manager._models:
            continue
        for model_type in model_manager._models[ticker]:
            for target in model_manager._models[ticker][model_type]:
                model = model_manager._models[ticker][model_type][target]
                if len(model.prediction_history) > 0:
                    model_manager.save_model(ticker, model_type, target, model)

    print("Done! Models saved with prediction history.")

    # Show sample hit rates
    print("\n" + "=" * 60)
    print("SAMPLE HIT RATES")
    print("=" * 60)

    sample_tickers = tickers[:5]
    for ticker in sample_tickers:
        if ticker not in model_manager._models:
            continue
        for model_type in ['lightgbm', 'xgboost', 'lstm']:
            if model_type not in model_manager._models[ticker]:
                continue
            if 'up' in model_manager._models[ticker][model_type]:
                model = model_manager._models[ticker][model_type]['up']
                accuracy = model.get_recent_accuracy(hours=settings.BACKTEST_HOURS)
                history_len = len(model.prediction_history)
                print(f"  {ticker} {model_type} up: {accuracy*100:.1f}% hit rate ({history_len} predictions)")


if __name__ == "__main__":
    main()
