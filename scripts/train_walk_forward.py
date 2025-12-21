#!/usr/bin/env python3
"""
Walk-Forward Training and Validation

Implements rolling window training to prevent data leakage and provide
realistic out-of-sample performance estimates.

Walk-Forward Process:
1. Split data into rolling windows
2. Train on each window
3. Test on the next period (out-of-sample)
4. Aggregate all OOS results

Example with 60 days data:
- Window 1: Train Day 1-40,  Test Day 41-45
- Window 2: Train Day 6-45,  Test Day 46-50
- Window 3: Train Day 11-50, Test Day 51-55
- Window 4: Train Day 16-55, Test Day 56-60

Usage:
    python scripts/train_walk_forward.py
    python scripts/train_walk_forward.py --train-days 40 --test-days 5 --step-days 5
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=True)

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select
from tqdm import tqdm

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator

# Import model classes directly for Walk-Forward (avoid saving to disk)
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel

warnings.filterwarnings("ignore")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )


def get_tickers_with_data(min_bars: int = 2000) -> List[str]:
    """Get tickers that have sufficient historical data."""
    try:
        from sqlalchemy import text

        with get_db() as db:
            result = db.execute(
                text(
                    f"SELECT symbol, COUNT(*) as bar_count FROM minute_bars "
                    f"GROUP BY symbol HAVING bar_count >= {min_bars} "
                    f"ORDER BY bar_count DESC"
                )
            )
            tickers = [row[0] for row in result]
            return tickers

    except Exception as e:
        logger.error(f"Failed to get tickers: {e}")
        return []


def load_ticker_data(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
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

            return df

    except Exception as e:
        logger.error(f"{ticker}: Failed to load data: {e}")
        return None


def prepare_features_labels(
    df: pd.DataFrame,
    feature_engineer: FeatureEngineer,
    label_generator: LabelGenerator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime]]:
    """Prepare features and labels from dataframe."""

    # Compute features
    features_df = feature_engineer.compute_features(df)
    feature_names = feature_engineer.get_feature_names()

    # Generate labels
    labels_up = []
    labels_down = []
    timestamps = []

    for idx in range(len(df) - settings.PREDICTION_HORIZON_MINUTES - 1):
        entry_time = df.iloc[idx]["timestamp"]
        entry_price = df.iloc[idx]["close"]

        labels = label_generator.generate_labels(df, entry_time, entry_price)
        labels_up.append(labels["label_up"])
        labels_down.append(labels["label_down"])
        timestamps.append(entry_time)

    # Align features and labels
    X = features_df[feature_names].values[:len(labels_up)]
    y_up = np.array(labels_up)
    y_down = np.array(labels_down)

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_up = y_up[valid_mask]
    y_down = y_down[valid_mask]
    timestamps = [t for t, v in zip(timestamps, valid_mask) if v]

    return X, y_up, y_down, timestamps


def get_window_indices(
    timestamps: List[datetime],
    train_days: int,
    test_days: int,
    step_days: int
) -> List[Dict]:
    """
    Calculate train/test window indices.

    Returns list of dicts with train_start, train_end, test_start, test_end indices.
    """
    if not timestamps:
        return []

    start_date = min(timestamps).date()
    end_date = max(timestamps).date()
    total_days = (end_date - start_date).days + 1

    windows = []
    window_start = 0

    while True:
        train_end_date = start_date + timedelta(days=window_start + train_days)
        test_end_date = train_end_date + timedelta(days=test_days)

        if test_end_date > end_date:
            break

        # Find indices
        train_start_date = start_date + timedelta(days=window_start)

        train_indices = [
            i for i, t in enumerate(timestamps)
            if train_start_date <= t.date() < train_end_date
        ]

        test_indices = [
            i for i, t in enumerate(timestamps)
            if train_end_date <= t.date() < test_end_date
        ]

        if train_indices and test_indices:
            windows.append({
                'train_start': min(train_indices),
                'train_end': max(train_indices) + 1,
                'test_start': min(test_indices),
                'test_end': max(test_indices) + 1,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'test_start_date': train_end_date,
                'test_end_date': test_end_date,
            })

        window_start += step_days

    return windows


def train_and_predict_window(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    ticker: str = 'TEMP',
    model_type: str = 'lightgbm'
) -> np.ndarray:
    """Train model on window and predict on test set."""

    try:
        if model_type == 'xgboost':
            model = XGBoostModel(ticker=ticker, target='up', use_gpu=settings.USE_GPU)
        else:
            model = LightGBMModel(ticker=ticker, target='up', use_gpu=settings.USE_GPU)

        # Train
        model.train(X_train, y_train)

        # Predict
        if model.is_trained:
            probs = model.predict_proba(X_test)
            return probs
        else:
            return np.zeros(len(X_test))
    except Exception as e:
        logger.debug(f"Training failed: {e}")
        return np.zeros(len(X_test))


def simulate_trades(
    df: pd.DataFrame,
    test_indices: List[int],
    probabilities: np.ndarray,
    threshold: float = 0.10,
    target_pct: float = 1.0,
    horizon_bars: int = 12,
    commission: float = 0.2
) -> List[Dict]:
    """Simulate trades on test period."""
    trades = []

    i = 0
    while i < len(test_indices):
        idx = test_indices[i]
        prob = probabilities[i]

        if prob < threshold:
            i += 1
            continue

        if idx + horizon_bars >= len(df):
            break

        entry_price = df.iloc[idx]['close']
        entry_time = df.iloc[idx]['timestamp']

        exit_idx = None
        exit_price = None
        exit_reason = None

        for j in range(1, min(horizon_bars + 1, len(df) - idx)):
            bar = df.iloc[idx + j]
            high_return = (bar['high'] - entry_price) / entry_price * 100

            if high_return >= target_pct:
                exit_price = entry_price * (1 + target_pct / 100)
                exit_reason = 'target_hit'
                exit_idx = idx + j
                break

        if exit_idx is None:
            exit_idx = min(idx + horizon_bars, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'time_limit'

        gross_pnl = (exit_price - entry_price) / entry_price * 100
        net_pnl = gross_pnl - commission

        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'probability': prob,
            'net_pnl': net_pnl,
        })

        # Find next test index after exit
        exit_time = df.iloc[exit_idx]['timestamp']
        while i < len(test_indices) and test_indices[i] <= exit_idx:
            i += 1

    return trades


def run_walk_forward(
    ticker: str,
    df: pd.DataFrame,
    train_days: int = 40,
    test_days: int = 5,
    step_days: int = 5,
    model_type: str = 'lightgbm'
) -> Dict:
    """Run walk-forward analysis for a single ticker."""

    feature_engineer = FeatureEngineer(ticker=ticker)
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

    # Prepare all data
    X, y_up, y_down, timestamps = prepare_features_labels(
        df, feature_engineer, label_generator
    )

    if len(X) < 500:
        return {'ticker': ticker, 'success': False, 'reason': 'insufficient_data'}

    # Get windows
    windows = get_window_indices(timestamps, train_days, test_days, step_days)

    if not windows:
        return {'ticker': ticker, 'success': False, 'reason': 'no_windows'}

    all_trades = []
    window_results = []

    for w_idx, window in enumerate(windows):
        # Split data
        X_train = X[window['train_start']:window['train_end']]
        y_train = y_up[window['train_start']:window['train_end']]
        X_test = X[window['test_start']:window['test_end']]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Train and predict
        probs = train_and_predict_window(X_train, y_train, X_test, ticker, model_type)

        # Get test indices in original df
        test_indices = list(range(window['test_start'], window['test_end']))

        # Simulate trades
        trades = simulate_trades(
            df, test_indices, probs,
            threshold=settings.PROBABILITY_THRESHOLD,
            target_pct=settings.TARGET_PERCENT,
            horizon_bars=12,
            commission=0.2
        )

        all_trades.extend(trades)

        # Window stats
        if trades:
            wins = sum(1 for t in trades if t['net_pnl'] > 0)
            window_results.append({
                'window': w_idx + 1,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'trades': len(trades),
                'win_rate': wins / len(trades) * 100,
                'total_pnl': sum(t['net_pnl'] for t in trades),
            })

    return {
        'ticker': ticker,
        'success': True,
        'windows': len(windows),
        'window_results': window_results,
        'all_trades': all_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Training")
    parser.add_argument('--train-days', type=int, default=40, help='Training window size in days')
    parser.add_argument('--test-days', type=int, default=5, help='Test window size in days')
    parser.add_argument('--step-days', type=int, default=5, help='Step size between windows')
    parser.add_argument('--total-days', type=int, default=60, help='Total days of data to use')
    parser.add_argument('--model-type', choices=['xgboost', 'lightgbm'], default='lightgbm')
    parser.add_argument('--max-tickers', type=int, help='Maximum tickers to process')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    setup_logging(args.verbose)

    print("=" * 70)
    print("WALK-FORWARD TRAINING AND VALIDATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total data:      {args.total_days} days")
    print(f"  Training window: {args.train_days} days")
    print(f"  Test window:     {args.test_days} days")
    print(f"  Step size:       {args.step_days} days")
    print(f"  Model type:      {args.model_type}")

    # Calculate expected windows
    expected_windows = (args.total_days - args.train_days) // args.step_days
    print(f"  Expected windows: ~{expected_windows}")
    print()

    # Get tickers
    tickers = get_tickers_with_data(min_bars=2000)
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]

    print(f"Processing {len(tickers)} tickers...\n")

    all_results = []
    all_trades = []

    pbar = tqdm(tickers, desc="Walk-Forward", ncols=100)

    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")

        df = load_ticker_data(ticker, days=args.total_days)
        if df is None or len(df) < 1000:
            continue

        result = run_walk_forward(
            ticker, df,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            model_type=args.model_type
        )

        if result['success']:
            all_results.append(result)
            all_trades.extend(result['all_trades'])

            # Update progress
            if result['all_trades']:
                wins = sum(1 for t in result['all_trades'] if t['net_pnl'] > 0)
                wr = wins / len(result['all_trades']) * 100
                pbar.set_postfix({'trades': len(all_trades), 'WR': f'{wr:.0f}%'})

    # Summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS (All Out-of-Sample)")
    print("=" * 70)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        total_trades = len(trades_df)
        wins = (trades_df.net_pnl > 0).sum()
        win_rate = wins / total_trades * 100
        avg_pnl = trades_df.net_pnl.mean()
        total_return = trades_df.net_pnl.sum()
        target_hits = (trades_df.exit_reason == 'target_hit').sum()
        target_rate = target_hits / total_trades * 100

        print(f"\nTotal OOS Trades:    {total_trades:,}")
        print(f"Win Rate:            {win_rate:.1f}%")
        print(f"Avg P&L per trade:   {avg_pnl:.3f}%")
        print(f"Total Return:        {total_return:.2f}%")
        print(f"Target Hit Rate:     {target_rate:.1f}%")

        # Profit factor
        total_profit = trades_df[trades_df.net_pnl > 0].net_pnl.sum()
        total_loss = abs(trades_df[trades_df.net_pnl <= 0].net_pnl.sum())
        pf = total_profit / total_loss if total_loss > 0 else 0
        print(f"Profit Factor:       {pf:.2f}")

        # Per-window breakdown
        print("\n--- Per-Window Breakdown ---")
        window_stats = []
        for r in all_results:
            for w in r.get('window_results', []):
                window_stats.append(w)

        if window_stats:
            ws_df = pd.DataFrame(window_stats)
            print(f"{'Window':<10} {'Trades':<10} {'Win Rate':<12} {'Total P&L':<12}")
            print("-" * 46)
            for w in range(1, ws_df.window.max() + 1):
                w_data = ws_df[ws_df.window == w]
                if len(w_data) > 0:
                    trades = w_data.trades.sum()
                    wr = (w_data.trades * w_data.win_rate).sum() / trades if trades > 0 else 0
                    pnl = w_data.total_pnl.sum()
                    print(f"Window {w:<3} {trades:<10} {wr:<11.1f}% {pnl:<+11.2f}%")

        # Interpretation
        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if avg_pnl > 0:
            print("\n[OK] Walk-Forward shows POSITIVE expected returns")
            print("     Model has genuine predictive power!")
        else:
            print("\n[WARNING] Walk-Forward shows NEGATIVE expected returns")
            print("          Model may still be overfitting within windows")

        if win_rate > 50:
            print(f"\n[OK] Win rate ({win_rate:.1f}%) is above 50%")
        else:
            print(f"\n[CONCERN] Win rate ({win_rate:.1f}%) is below 50%")

        # Per-ticker analysis
        print("\n" + "=" * 70)
        print("TOP PERFORMING TICKERS (by Win Rate)")
        print("=" * 70)

        ticker_stats = []
        for r in all_results:
            ticker_trades = r.get('all_trades', [])
            if len(ticker_trades) >= 3:  # Minimum 3 trades
                t_wins = sum(1 for t in ticker_trades if t['net_pnl'] > 0)
                t_wr = t_wins / len(ticker_trades) * 100
                t_pnl = sum(t['net_pnl'] for t in ticker_trades)
                t_avg_pnl = t_pnl / len(ticker_trades)

                # Profit factor
                t_profit = sum(t['net_pnl'] for t in ticker_trades if t['net_pnl'] > 0)
                t_loss = abs(sum(t['net_pnl'] for t in ticker_trades if t['net_pnl'] <= 0))
                t_pf = t_profit / t_loss if t_loss > 0 else float('inf')

                ticker_stats.append({
                    'ticker': r['ticker'],
                    'trades': len(ticker_trades),
                    'wins': t_wins,
                    'win_rate': t_wr,
                    'avg_pnl': t_avg_pnl,
                    'total_pnl': t_pnl,
                    'profit_factor': t_pf,
                })

        # Sort by win_rate descending
        ticker_stats_sorted = sorted(ticker_stats, key=lambda x: x['win_rate'], reverse=True)

        print(f"\n{'Ticker':<8} {'Trades':<8} {'Win Rate':<10} {'Avg P&L':<10} {'Total P&L':<12} {'PF':<8}")
        print("-" * 60)

        # Show top 20
        for ts in ticker_stats_sorted[:20]:
            pf_str = f"{ts['profit_factor']:.2f}" if ts['profit_factor'] < float('inf') else "∞"
            print(f"{ts['ticker']:<8} {ts['trades']:<8} {ts['win_rate']:<9.1f}% {ts['avg_pnl']:<+9.3f}% {ts['total_pnl']:<+11.2f}% {pf_str:<8}")

        # Count profitable tickers
        profitable_tickers = [ts for ts in ticker_stats if ts['avg_pnl'] > 0]
        print(f"\n--- Summary ---")
        print(f"Tickers with >= 3 trades: {len(ticker_stats)}")
        print(f"Profitable tickers (avg P&L > 0): {len(profitable_tickers)}")
        print(f"Top ticker Win Rate: {ticker_stats_sorted[0]['win_rate']:.1f}%" if ticker_stats_sorted else "N/A")

        # If we only used profitable tickers
        if profitable_tickers:
            pt_trades = sum(ts['trades'] for ts in profitable_tickers)
            pt_wins = sum(ts['wins'] for ts in profitable_tickers)
            pt_pnl = sum(ts['total_pnl'] for ts in profitable_tickers)
            print(f"\n--- If ONLY using profitable tickers ---")
            print(f"Trades: {pt_trades}")
            print(f"Win Rate: {pt_wins/pt_trades*100:.1f}%")
            print(f"Total Return: {pt_pnl:+.2f}%")

        # High win rate tickers (>55%)
        high_wr_tickers = [ts for ts in ticker_stats if ts['win_rate'] >= 55]
        if high_wr_tickers:
            hwr_trades = sum(ts['trades'] for ts in high_wr_tickers)
            hwr_wins = sum(ts['wins'] for ts in high_wr_tickers)
            hwr_pnl = sum(ts['total_pnl'] for ts in high_wr_tickers)
            print(f"\n--- If ONLY using >55% WR tickers ---")
            print(f"Tickers: {len(high_wr_tickers)}")
            print(f"Trades: {hwr_trades}")
            print(f"Win Rate: {hwr_wins/hwr_trades*100:.1f}%")
            print(f"Total Return: {hwr_pnl:+.2f}%")

        # Save results
        output_path = project_root / "data" / "walk_forward_results.json"
        results_summary = {
            'config': {
                'train_days': args.train_days,
                'test_days': args.test_days,
                'step_days': args.step_days,
                'model_type': args.model_type,
            },
            'summary': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_return': total_return,
                'profit_factor': pf,
            },
            'per_ticker': ticker_stats_sorted,
            'profitable_tickers': [ts['ticker'] for ts in profitable_tickers],
            'high_winrate_tickers': [ts['ticker'] for ts in high_wr_tickers] if high_wr_tickers else [],
            'tickers_processed': len(all_results),
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    else:
        print("\nNo trades generated!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
