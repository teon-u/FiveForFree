#!/usr/bin/env python3
"""
Populate model_performance table with backtest results.

This script:
1. Loads all trained models
2. Runs backtests on historical data
3. Calculates performance metrics
4. Stores results in the model_performance database table
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import select

from config.settings import settings
from src.utils.database import get_db, init_db, MinuteBar as DBMinuteBar, ModelPerformance
from src.models.model_manager import ModelManager
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator


def calculate_trading_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    prices: np.ndarray,
    target_percent: float = 1.0,
) -> Dict:
    """
    Calculate trading-related metrics from predictions.

    Args:
        predictions: Binary predictions (0 or 1)
        actuals: Actual outcomes (0 or 1)
        prices: Price series for return calculation
        target_percent: Target movement percentage

    Returns:
        Dictionary of trading metrics
    """
    if len(predictions) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': None,
            'avg_pnl_percent': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
        }

    # Count trades (predictions == 1)
    trade_mask = predictions == 1
    total_trades = int(trade_mask.sum())

    if total_trades == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': None,
            'avg_pnl_percent': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
        }

    # Winning trades (prediction == 1 and actual == 1)
    winning_trades = int((trade_mask & (actuals == 1)).sum())
    win_rate = winning_trades / total_trades if total_trades > 0 else None

    # Calculate returns for each trade
    returns = []
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        if pred == 1:
            # Simulated return based on actual outcome
            if actual == 1:
                returns.append(target_percent / 100)  # Hit target
            else:
                returns.append(-target_percent / 200)  # Stop loss at half target

    returns = np.array(returns)
    avg_pnl_percent = float(returns.mean() * 100) if len(returns) > 0 else None

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1 and returns.std() > 0:
        daily_factor = 252 / len(returns)  # Rough annualization
        sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(daily_factor))
    else:
        sharpe_ratio = None

    # Max drawdown
    if len(returns) > 0:
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = None

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_pnl_percent': avg_pnl_percent,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }


def evaluate_model_performance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    prices: np.ndarray,
    model_type: str,
    symbol: str,
    target: str,
    period_start: datetime,
    period_end: datetime,
) -> Optional[ModelPerformance]:
    """
    Evaluate a single model and create ModelPerformance record.

    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test labels
        prices: Price series
        model_type: Type of model
        symbol: Stock ticker
        target: 'up' or 'down'
        period_start: Start of evaluation period
        period_end: End of evaluation period

    Returns:
        ModelPerformance object or None if evaluation fails
    """
    try:
        # Get predictions
        probs = model.predict_proba(X_test)
        preds = (probs >= 0.5).astype(int)

        # Basic metrics
        total_predictions = len(y_test)
        correct_predictions = int((preds == y_test).sum())
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Confidence metrics
        correct_mask = preds == y_test
        avg_prob_correct = float(probs[correct_mask].mean()) if correct_mask.any() else None
        avg_prob_incorrect = float(probs[~correct_mask].mean()) if (~correct_mask).any() else None

        # Trading metrics
        trading_metrics = calculate_trading_metrics(preds, y_test, prices, settings.TARGET_PERCENT)

        # Direction-specific counts
        if target == 'up':
            up_predictions = total_predictions
            up_accuracy = accuracy
            down_predictions = 0
            down_accuracy = None
        else:
            up_predictions = 0
            up_accuracy = None
            down_predictions = total_predictions
            down_accuracy = accuracy

        # Additional metrics as JSON
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        metrics_json = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'positive_rate': float(y_test.mean()),
            'prediction_rate': float(preds.mean()),
            'target_direction': target,
        }

        # Create performance record
        performance = ModelPerformance(
            model_type=model_type,
            symbol=symbol,
            evaluation_date=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            up_predictions=up_predictions,
            down_predictions=down_predictions,
            up_accuracy=up_accuracy,
            down_accuracy=down_accuracy,
            avg_prob_correct=avg_prob_correct,
            avg_prob_incorrect=avg_prob_incorrect,
            total_trades=trading_metrics['total_trades'],
            winning_trades=trading_metrics['winning_trades'],
            win_rate=trading_metrics['win_rate'],
            avg_pnl_percent=trading_metrics['avg_pnl_percent'],
            sharpe_ratio=trading_metrics['sharpe_ratio'],
            max_drawdown=trading_metrics['max_drawdown'],
            model_version=getattr(model, 'version', '1.0'),
            training_samples=getattr(model, 'training_samples', None),
            feature_count=X_test.shape[1] if len(X_test.shape) > 1 else None,
            metrics_json=metrics_json,
        )

        return performance

    except Exception as e:
        print(f"    Error evaluating {model_type}/{target}: {e}")
        return None


def main():
    print("=" * 70)
    print("POPULATE MODEL_PERFORMANCE TABLE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize database
    init_db()

    # Load models
    model_manager = ModelManager()
    loaded_count = model_manager.load_all_models()
    print(f"Loaded {loaded_count} models from {model_manager.get_summary()['total_tickers']} tickers")

    # Get all tickers
    tickers = model_manager.get_tickers()
    print(f"Evaluating {len(tickers)} tickers...")
    print()

    # Initialize processors
    feature_engineer = FeatureEngineer()
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    total_records = 0
    successful_tickers = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            # Load data (last 30 days for evaluation)
            cutoff_date = datetime.now() - timedelta(days=30)
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

            if len(df) < 500:
                continue

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()

            # Generate labels
            labels_up = []
            labels_down = []
            for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
                entry_time = df.iloc[idx]["timestamp"]
                entry_price = df.iloc[idx]["close"]
                labels = label_generator.generate_labels(df, entry_time, entry_price)
                labels_up.append(labels["label_up"])
                labels_down.append(labels["label_down"])

            # Align features and labels
            X = features_df[feature_names].values[:len(labels_up)]
            y_up = np.array(labels_up)
            y_down = np.array(labels_down)
            prices = df['close'].values[:len(labels_up)]

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1)
            X = X[valid_mask]
            y_up = y_up[valid_mask]
            y_down = y_down[valid_mask]
            prices = prices[valid_mask]

            if len(X) < 100:
                continue

            # Use last 20% for testing
            test_size = int(len(X) * 0.2)
            if test_size < 50:
                continue

            X_test = X[-test_size:]
            y_up_test = y_up[-test_size:]
            y_down_test = y_down[-test_size:]
            prices_test = prices[-test_size:]

            period_start = df.iloc[-test_size]['timestamp']
            period_end = df.iloc[-1]['timestamp']

            print(f"[{i}/{len(tickers)}] {ticker}: {len(X_test)} test samples")

            ticker_records = 0

            # Evaluate each model for this ticker
            for target in ["up", "down"]:
                y_test = y_up_test if target == "up" else y_down_test

                for model_type in model_manager.MODEL_TYPES:
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

                        # Evaluate and create performance record
                        performance = evaluate_model_performance(
                            model=model,
                            X_test=X_test,
                            y_test=y_test,
                            prices=prices_test,
                            model_type=model_type,
                            symbol=ticker,
                            target=target,
                            period_start=period_start,
                            period_end=period_end,
                        )

                        if performance:
                            # Save to database
                            with get_db() as db:
                                db.add(performance)
                            ticker_records += 1

                    except Exception as e:
                        continue

            if ticker_records > 0:
                successful_tickers += 1
                total_records += ticker_records
                print(f"    Added {ticker_records} performance records")

        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tickers processed: {successful_tickers}/{len(tickers)}")
    print(f"Performance records added: {total_records}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show sample records
    if total_records > 0:
        print()
        print("=" * 70)
        print("SAMPLE RECORDS (Top 10 by Accuracy)")
        print("=" * 70)
        with get_db() as db:
            stmt = (
                select(ModelPerformance)
                .order_by(ModelPerformance.accuracy.desc())
                .limit(10)
            )
            records = db.execute(stmt).scalars().all()

            for rec in records:
                print(f"  {rec.symbol} {rec.model_type}: "
                      f"acc={rec.accuracy:.1%}, "
                      f"trades={rec.total_trades}, "
                      f"win_rate={rec.win_rate:.1%}" if rec.win_rate else "n/a")


if __name__ == "__main__":
    main()
