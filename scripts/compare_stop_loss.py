#!/usr/bin/env python3
"""
Stop Loss Strategy Comparison using Actual Model Predictions

Compares backtest results with different stop loss configurations:
1. No stop loss (time limit only)
2. -0.95% stop loss
3. -2.0% stop loss

Uses trained ML models for predictions instead of pseudo-momentum.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from loguru import logger

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


def simulate_with_stop_loss(
    ticker: str,
    df: pd.DataFrame,
    probabilities: np.ndarray,
    stop_loss_pct: float = None,  # None = no stop loss
    target_pct: float = 1.0,
    horizon_bars: int = 12,
    threshold: float = 0.10,
    commission: float = 0.2
) -> dict:
    """
    Simulate trades with specified stop loss.

    Args:
        ticker: Stock ticker
        df: DataFrame with OHLCV data
        probabilities: Model prediction probabilities
        stop_loss_pct: Stop loss percentage (None = no stop loss)
        target_pct: Target profit percentage
        horizon_bars: Max bars to hold
        threshold: Probability threshold for entry
        commission: Round-trip commission percentage

    Returns:
        Dictionary with simulation results
    """
    trades = []
    current_idx = 0
    total_bars = len(df)

    while current_idx < total_bars - horizon_bars:
        prob = probabilities[current_idx]

        if prob < threshold:
            current_idx += 1
            continue

        # Entry
        entry_price = df.iloc[current_idx]['close']
        entry_time = df.iloc[current_idx]['timestamp']

        exit_idx = None
        exit_price = None
        exit_reason = None

        # Simulate trade
        for j in range(current_idx + 1, min(current_idx + horizon_bars + 1, total_bars)):
            bar = df.iloc[j]
            high_return = (bar['high'] - entry_price) / entry_price * 100
            low_return = (bar['low'] - entry_price) / entry_price * 100

            # Check target hit
            if high_return >= target_pct:
                exit_price = entry_price * (1 + target_pct / 100)
                exit_reason = 'target_hit'
                exit_idx = j
                break

            # Check stop loss (if enabled)
            if stop_loss_pct is not None and low_return <= -stop_loss_pct:
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                exit_reason = 'stop_loss'
                exit_idx = j
                break

        # Time limit exit
        if exit_idx is None:
            exit_idx = min(current_idx + horizon_bars, total_bars - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'time_limit'

        # Calculate P&L
        gross_pnl = (exit_price - entry_price) / entry_price * 100
        net_pnl = gross_pnl - commission

        trades.append({
            'ticker': ticker,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'probability': prob,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'duration_bars': exit_idx - current_idx
        })

        # Immediate re-entry
        current_idx = exit_idx + 1

    return trades


def analyze_trades(trades: list, label: str) -> dict:
    """Analyze trade results."""
    if not trades:
        return {'label': label, 'total_trades': 0}

    df = pd.DataFrame(trades)

    wins = df[df.net_pnl > 0]
    losses = df[df.net_pnl <= 0]

    target_hits = (df.exit_reason == 'target_hit').sum()
    stop_losses = (df.exit_reason == 'stop_loss').sum()
    time_limits = (df.exit_reason == 'time_limit').sum()

    total_profit = wins.net_pnl.sum() if len(wins) > 0 else 0
    total_loss = abs(losses.net_pnl.sum()) if len(losses) > 0 else 0

    return {
        'label': label,
        'total_trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(df) * 100,
        'avg_pnl': df.net_pnl.mean(),
        'avg_win': wins.net_pnl.mean() if len(wins) > 0 else 0,
        'avg_loss': losses.net_pnl.mean() if len(losses) > 0 else 0,
        'total_return': df.net_pnl.sum(),
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
        'target_hits': target_hits,
        'stop_losses': stop_losses,
        'time_limits': time_limits,
        'target_hit_rate': target_hits / len(df) * 100,
        'avg_duration': df.duration_bars.mean(),
        'max_drawdown': (df.net_pnl.cumsum().cummax() - df.net_pnl.cumsum()).max()
    }


def main():
    print("=" * 70)
    print("Stop Loss Strategy Comparison (Using Actual Model Predictions)")
    print("=" * 70)

    # Load models
    print("\nLoading trained models...")
    model_manager = ModelManager()
    loaded_count = model_manager.load_all_models()
    print(f"Loaded {loaded_count} models")

    # Get tickers with trained models
    tickers = model_manager.get_tickers()
    print(f"Tickers with models: {len(tickers)}")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Stop loss configurations to test
    configs = [
        {'name': 'No Stop Loss', 'stop_loss': None},
        {'name': 'Stop Loss -0.95%', 'stop_loss': 0.95},
        {'name': 'Stop Loss -2.0%', 'stop_loss': 2.0},
    ]

    # Collect all trades for each configuration
    all_results = {cfg['name']: [] for cfg in configs}

    processed = 0
    skipped = 0

    for ticker in tickers:
        try:
            # Load data
            df = load_ticker_data(ticker, days=60)
            if len(df) < 500:
                skipped += 1
                continue

            # Generate features
            features_df = feature_engineer.compute_features(df)
            feature_names = feature_engineer.get_feature_names()
            X = features_df[feature_names].values

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1)
            X_valid = X[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            if len(X_valid) < 100:
                skipped += 1
                continue

            # Get model predictions (use best available model)
            probabilities = np.zeros(len(df))

            model = None
            for model_type in ['lightgbm', 'xgboost', 'lstm', 'transformer']:
                if ticker in model_manager._models:
                    if model_type in model_manager._models[ticker]:
                        if 'up' in model_manager._models[ticker][model_type]:
                            m = model_manager._models[ticker][model_type]['up']
                            if m.is_trained and m._model is not None:
                                model = m
                                break

            if model is None:
                skipped += 1
                continue

            # Predict
            try:
                probs = model.predict_proba(X_valid)
                for i, idx in enumerate(valid_indices):
                    probabilities[idx] = probs[i]
            except Exception as e:
                skipped += 1
                continue

            # Simulate with each stop loss configuration
            for cfg in configs:
                trades = simulate_with_stop_loss(
                    ticker=ticker,
                    df=df,
                    probabilities=probabilities,
                    stop_loss_pct=cfg['stop_loss'],
                    target_pct=1.0,
                    horizon_bars=12,
                    threshold=settings.PROBABILITY_THRESHOLD,
                    commission=0.2
                )
                all_results[cfg['name']].extend(trades)

            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} tickers...")

        except Exception as e:
            skipped += 1
            continue

    print(f"\nProcessed: {processed} tickers, Skipped: {skipped}")

    # Analyze results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    results_summary = []
    for cfg in configs:
        trades = all_results[cfg['name']]
        analysis = analyze_trades(trades, cfg['name'])
        results_summary.append(analysis)

    # Print comparison table
    print(f"\n{'Metric':<25} | {'No Stop Loss':>15} | {'SL -0.95%':>15} | {'SL -2.0%':>15}")
    print("-" * 75)

    metrics = [
        ('Total Trades', 'total_trades', '{:,.0f}'),
        ('Win Rate (%)', 'win_rate', '{:.1f}'),
        ('Avg P&L (%)', 'avg_pnl', '{:.3f}'),
        ('Avg Win (%)', 'avg_win', '{:.3f}'),
        ('Avg Loss (%)', 'avg_loss', '{:.3f}'),
        ('Total Return (%)', 'total_return', '{:.2f}'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Target Hits', 'target_hits', '{:,.0f}'),
        ('Stop Losses', 'stop_losses', '{:,.0f}'),
        ('Time Limits', 'time_limits', '{:,.0f}'),
        ('Target Hit Rate (%)', 'target_hit_rate', '{:.1f}'),
        ('Avg Duration (bars)', 'avg_duration', '{:.1f}'),
        ('Max Drawdown (%)', 'max_drawdown', '{:.2f}'),
    ]

    for metric_name, key, fmt in metrics:
        values = []
        for r in results_summary:
            if r['total_trades'] == 0:
                values.append('N/A')
            else:
                values.append(fmt.format(r.get(key, 0)))
        print(f"{metric_name:<25} | {values[0]:>15} | {values[1]:>15} | {values[2]:>15}")

    # Winner analysis
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    if all(r['total_trades'] > 0 for r in results_summary):
        # Find best by profit factor
        best_pf = max(results_summary, key=lambda x: x.get('profit_factor', 0))
        print(f"\nBest Profit Factor: {best_pf['label']} ({best_pf['profit_factor']:.2f})")

        # Find best by total return
        best_return = max(results_summary, key=lambda x: x.get('total_return', 0))
        print(f"Best Total Return: {best_return['label']} ({best_return['total_return']:.2f}%)")

        # Find best by win rate
        best_wr = max(results_summary, key=lambda x: x.get('win_rate', 0))
        print(f"Best Win Rate: {best_wr['label']} ({best_wr['win_rate']:.1f}%)")

        # Risk analysis
        print("\n--- Risk Analysis ---")
        for r in results_summary:
            if r['total_trades'] > 0:
                risk_reward = abs(r['avg_win'] / r['avg_loss']) if r['avg_loss'] != 0 else 0
                print(f"{r['label']}: Risk/Reward = 1:{risk_reward:.2f}, Max DD = {r['max_drawdown']:.2f}%")

    print("\n" + "=" * 70)
    print("Comparison complete!")


if __name__ == "__main__":
    main()
