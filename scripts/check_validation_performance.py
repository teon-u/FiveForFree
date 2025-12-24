#!/usr/bin/env python3
"""
Check Model Performance on Validation Set Only (Out-of-Sample)

This script tests models ONLY on the last 20% of data (validation set)
to get a realistic estimate of model performance without data leakage.
"""
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


def load_ticker_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """Load historical minute bar data for a ticker."""
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
            return pd.DataFrame()

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

        return df


def simulate_trades(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    target_pct: float = 1.0,
    horizon_bars: int = 12,
    threshold: float = 0.10,
    commission: float = 0.2
) -> list:
    """Simulate trades with immediate re-entry."""
    trades = []
    current_idx = 0
    total_bars = len(df)

    while current_idx < total_bars - horizon_bars:
        prob = probabilities[current_idx]

        if prob < threshold:
            current_idx += 1
            continue

        entry_price = df.iloc[current_idx]['close']

        exit_idx = None
        exit_price = None
        exit_reason = None

        for j in range(current_idx + 1, min(current_idx + horizon_bars + 1, total_bars)):
            bar = df.iloc[j]
            high_return = (bar['high'] - entry_price) / entry_price * 100

            if high_return >= target_pct:
                exit_price = entry_price * (1 + target_pct / 100)
                exit_reason = 'target_hit'
                exit_idx = j
                break

        if exit_idx is None:
            exit_idx = min(current_idx + horizon_bars, total_bars - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'time_limit'

        gross_pnl = (exit_price - entry_price) / entry_price * 100
        net_pnl = gross_pnl - commission

        trades.append({
            'net_pnl': net_pnl,
            'exit_reason': exit_reason
        })

        current_idx = exit_idx + 1

    return trades


def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION (Last 20% of Data Only)")
    print("=" * 70)
    print("\nThis test uses ONLY data that was NOT used for training.")
    print("This gives a realistic estimate of model performance.\n")

    # Load models
    model_manager = ModelManager()
    loaded_count = model_manager.load_all_models()
    print(f"Loaded {loaded_count} models")

    tickers = model_manager.get_tickers()
    print(f"Testing {len(tickers)} tickers...")

    feature_engineer = FeatureEngineer()

    # Results for different data splits
    results = {
        'train_set': {'trades': [], 'label': 'Train Set (First 80%)'},
        'val_set': {'trades': [], 'label': 'Validation Set (Last 20%)'},
    }

    processed = 0

    for ticker in tickers:
        try:
            df = load_ticker_data(ticker, days=60)
            if len(df) < 500:
                continue

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()
            X = features_df[feature_names].values

            valid_mask = ~np.isnan(X).any(axis=1)
            X_valid = X[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            if len(X_valid) < 100:
                continue

            # Get model predictions
            probabilities = np.zeros(len(df))

            model = None
            for model_type in ['lightgbm', 'xgboost']:
                if ticker in model_manager._models:
                    if model_type in model_manager._models[ticker]:
                        if 'up' in model_manager._models[ticker][model_type]:
                            m = model_manager._models[ticker][model_type]['up']
                            if m.is_trained and m._model is not None:
                                model = m
                                break

            if model is None:
                continue

            try:
                probs = model.predict_proba(X_valid)
                for i, idx in enumerate(valid_indices):
                    probabilities[idx] = probs[i]
            except:
                continue

            # Split data: 80% train, 20% validation
            n_total = len(df)
            n_train = int(n_total * 0.8)

            # Train set (first 80%)
            df_train = df.iloc[:n_train].reset_index(drop=True)
            probs_train = probabilities[:n_train]

            # Validation set (last 20%)
            df_val = df.iloc[n_train:].reset_index(drop=True)
            probs_val = probabilities[n_train:]

            # Simulate on each set
            train_trades = simulate_trades(df_train, probs_train)
            val_trades = simulate_trades(df_val, probs_val)

            results['train_set']['trades'].extend(train_trades)
            results['val_set']['trades'].extend(val_trades)

            processed += 1

        except Exception as e:
            continue

    print(f"\nProcessed: {processed} tickers")

    # Analyze results
    print("\n" + "=" * 70)
    print("COMPARISON: Train Set vs Validation Set (Out-of-Sample)")
    print("=" * 70)

    print(f"\n{'Metric':<25} | {'Train (80%)':<20} | {'Val (20%)':<20}")
    print("-" * 70)

    for key in ['train_set', 'val_set']:
        trades = results[key]['trades']
        if not trades:
            continue

        df_trades = pd.DataFrame(trades)
        n_trades = len(df_trades)
        wins = (df_trades.net_pnl > 0).sum()
        win_rate = wins / n_trades * 100 if n_trades > 0 else 0
        avg_pnl = df_trades.net_pnl.mean()
        total_return = df_trades.net_pnl.sum()
        target_hits = (df_trades.exit_reason == 'target_hit').sum()
        target_rate = target_hits / n_trades * 100 if n_trades > 0 else 0

        results[key]['stats'] = {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'target_rate': target_rate
        }

    # Print comparison
    if results['train_set'].get('stats') and results['val_set'].get('stats'):
        train = results['train_set']['stats']
        val = results['val_set']['stats']

        print(f"{'Total Trades':<25} | {train['n_trades']:>18,} | {val['n_trades']:>18,}")
        print(f"{'Win Rate (%)':<25} | {train['win_rate']:>18.1f} | {val['win_rate']:>18.1f}")
        print(f"{'Avg P&L (%)':<25} | {train['avg_pnl']:>18.3f} | {val['avg_pnl']:>18.3f}")
        print(f"{'Total Return (%)':<25} | {train['total_return']:>18.1f} | {val['total_return']:>18.1f}")
        print(f"{'Target Hit Rate (%)':<25} | {train['target_rate']:>18.1f} | {val['target_rate']:>18.1f}")

        # Calculate drop
        print("\n" + "=" * 70)
        print("PERFORMANCE DROP (Train → Validation)")
        print("=" * 70)

        wr_drop = train['win_rate'] - val['win_rate']
        pnl_drop = train['avg_pnl'] - val['avg_pnl']

        print(f"\nWin Rate Drop:    {wr_drop:+.1f}%p")
        print(f"Avg P&L Drop:     {pnl_drop:+.3f}%")

        if wr_drop > 10:
            print("\n⚠️  WARNING: Large performance drop detected!")
            print("   This suggests overfitting to training data.")
        elif wr_drop > 5:
            print("\n⚡ NOTICE: Moderate performance drop.")
            print("   Model may have some overfitting.")
        else:
            print("\n✅ Performance is relatively stable.")
            print("   Model generalizes reasonably well.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
