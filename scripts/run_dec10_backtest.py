#!/usr/bin/env python3
"""
December 10 Backtest Simulation

Realistic backtest that simulates:
1. Training models ONLY on data available as of December 10, 2025
2. Running investment simulation on December 10-20, 2025

This provides a true out-of-sample test of model performance.

Usage:
    python scripts/run_dec10_backtest.py
    python scripts/run_dec10_backtest.py --cutoff-date 2025-12-10 --test-days 10
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
from sqlalchemy import select, and_
from tqdm import tqdm

from config.settings import settings
from src.utils.database import get_db, MinuteBar as DBMinuteBar
from src.processor.feature_engineer import FeatureEngineer
from src.processor.label_generator import LabelGenerator

# Import model classes
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


def get_tickers_with_sufficient_data(cutoff_date: datetime, min_bars: int = 1000) -> List[str]:
    """Get tickers that have sufficient data before cutoff date."""
    try:
        from sqlalchemy import text

        with get_db() as db:
            result = db.execute(
                text(f"""
                    SELECT symbol, COUNT(*) as bar_count
                    FROM minute_bars
                    WHERE timestamp < '{cutoff_date.strftime('%Y-%m-%d')}'
                    GROUP BY symbol
                    HAVING bar_count >= {min_bars}
                    ORDER BY bar_count DESC
                """)
            )
            tickers = [row[0] for row in result]
            return tickers

    except Exception as e:
        logger.error(f"Failed to get tickers: {e}")
        return []


def load_training_data(ticker: str, cutoff_date: datetime) -> Optional[pd.DataFrame]:
    """Load historical data UP TO cutoff date (training data)."""
    try:
        with get_db() as db:
            stmt = (
                select(DBMinuteBar)
                .where(DBMinuteBar.symbol == ticker)
                .where(DBMinuteBar.timestamp < cutoff_date)
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
        logger.error(f"{ticker}: Failed to load training data: {e}")
        return None


def load_test_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Load historical data for test period (cutoff to end_date)."""
    try:
        with get_db() as db:
            stmt = (
                select(DBMinuteBar)
                .where(DBMinuteBar.symbol == ticker)
                .where(and_(
                    DBMinuteBar.timestamp >= start_date,
                    DBMinuteBar.timestamp <= end_date
                ))
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
        logger.error(f"{ticker}: Failed to load test data: {e}")
        return None


def prepare_features_labels(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime], List[str]]:
    """Prepare features and labels from dataframe."""

    feature_engineer = FeatureEngineer(ticker=ticker)
    label_generator = LabelGenerator(
        target_percent=settings.TARGET_PERCENT,
        prediction_horizon_minutes=settings.PREDICTION_HORIZON_MINUTES,
    )

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

    return X, y_up, y_down, timestamps, feature_names


def prepare_test_features(
    df: pd.DataFrame,
    ticker: str
) -> Tuple[np.ndarray, List[datetime], pd.DataFrame]:
    """Prepare features for test data (no labels needed for simulation)."""

    feature_engineer = FeatureEngineer(ticker=ticker)

    # Compute features
    features_df = feature_engineer.compute_features(df)
    feature_names = feature_engineer.get_feature_names()

    X = features_df[feature_names].values
    timestamps = df['timestamp'].tolist()[:len(X)]

    # Remove NaN rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    timestamps = [t for t, v in zip(timestamps, valid_mask) if v]
    df_filtered = df.iloc[:len(valid_mask)][valid_mask].reset_index(drop=True)

    return X, timestamps, df_filtered


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ticker: str,
    model_type: str = 'lightgbm'
):
    """Train model on training data."""

    try:
        if model_type == 'xgboost':
            model = XGBoostModel(ticker=ticker, target='up', use_gpu=settings.USE_GPU)
        else:
            model = LightGBMModel(ticker=ticker, target='up', use_gpu=settings.USE_GPU)

        model.train(X_train, y_train)

        if model.is_trained:
            return model
        else:
            return None

    except Exception as e:
        logger.debug(f"Training failed for {ticker}: {e}")
        return None


def simulate_trades(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    timestamps: List[datetime],
    threshold: float = 0.60,
    target_pct: float = 0.5,
    stop_loss_pct: float = 0.5,
    horizon_bars: int = 12,
    commission: float = 0.2
) -> List[Dict]:
    """Simulate trades on test period with immediate re-entry."""

    trades = []

    if len(df) == 0 or len(probabilities) == 0:
        return trades

    i = 0
    while i < len(probabilities):
        prob = probabilities[i]

        if prob < threshold:
            i += 1
            continue

        if i + horizon_bars >= len(df):
            break

        entry_price = df.iloc[i]['close']
        entry_time = timestamps[i]

        exit_idx = None
        exit_price = None
        exit_reason = None

        # Simulate trade
        for j in range(1, min(horizon_bars + 1, len(df) - i)):
            bar = df.iloc[i + j]
            high_return = (bar['high'] - entry_price) / entry_price * 100
            low_return = (bar['low'] - entry_price) / entry_price * 100

            # Check target first
            if high_return >= target_pct:
                exit_price = entry_price * (1 + target_pct / 100)
                exit_reason = 'target_hit'
                exit_idx = i + j
                break

            # Check stop loss
            if low_return <= -stop_loss_pct:
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                exit_reason = 'stop_loss'
                exit_idx = i + j
                break

        # Time limit exit
        if exit_idx is None:
            exit_idx = min(i + horizon_bars, len(df) - 1)
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
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
        })

        # Immediate re-entry: Move to bar right after exit
        i = exit_idx + 1

    return trades


def run_backtest(
    cutoff_date: datetime,
    test_end_date: datetime,
    model_type: str = 'lightgbm',
    max_tickers: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """Run full backtest simulation."""

    print("=" * 70)
    print("DECEMBER 10 BACKTEST SIMULATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training cutoff:  {cutoff_date.strftime('%Y-%m-%d')} (data before this)")
    print(f"  Test period:      {cutoff_date.strftime('%Y-%m-%d')} ~ {test_end_date.strftime('%Y-%m-%d')}")
    print(f"  Model type:       {model_type}")
    print(f"  Target profit:    {settings.TARGET_PERCENT}%")
    print(f"  Stop loss:        {settings.STOP_LOSS_PERCENT}%")
    print(f"  Threshold:        {settings.PROBABILITY_THRESHOLD}")
    print()

    # Get tickers with sufficient training data
    tickers = get_tickers_with_sufficient_data(cutoff_date, min_bars=500)
    if max_tickers:
        tickers = tickers[:max_tickers]

    print(f"Processing {len(tickers)} tickers with sufficient data...\n")

    all_trades = []
    ticker_results = []

    pbar = tqdm(tickers, desc="Backtest", ncols=100)

    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")

        # Load training data (before cutoff)
        train_df = load_training_data(ticker, cutoff_date)
        if train_df is None or len(train_df) < 500:
            continue

        # Load test data (cutoff to end)
        test_df = load_test_data(ticker, cutoff_date, test_end_date)
        if test_df is None or len(test_df) < 50:
            continue

        try:
            # Prepare training data
            X_train, y_up, y_down, train_ts, feature_names = prepare_features_labels(train_df, ticker)

            if len(X_train) < 200:
                continue

            # Train model
            model = train_model(X_train, y_up, ticker, model_type)
            if model is None:
                continue

            # Prepare test data
            X_test, test_timestamps, test_df_filtered = prepare_test_features(test_df, ticker)

            if len(X_test) < 10:
                continue

            # Predict
            probs = model.predict_proba(X_test)

            # Simulate trades
            trades = simulate_trades(
                test_df_filtered,
                probs,
                test_timestamps,
                threshold=settings.PROBABILITY_THRESHOLD,
                target_pct=settings.TARGET_PERCENT,
                stop_loss_pct=settings.STOP_LOSS_PERCENT,
                horizon_bars=12,
                commission=0.2
            )

            # Record trades with ticker
            for trade in trades:
                trade['ticker'] = ticker
                all_trades.append(trade)

            if trades:
                wins = sum(1 for t in trades if t['net_pnl'] > 0)
                ticker_results.append({
                    'ticker': ticker,
                    'trades': len(trades),
                    'wins': wins,
                    'win_rate': wins / len(trades) * 100,
                    'total_pnl': sum(t['net_pnl'] for t in trades),
                    'avg_pnl': sum(t['net_pnl'] for t in trades) / len(trades),
                })

                # Update progress
                pbar.set_postfix({
                    'trades': len(all_trades),
                    'WR': f'{wins/len(trades)*100:.0f}%'
                })

        except Exception as e:
            if verbose:
                logger.debug(f"{ticker}: Error - {e}")
            continue

    return {
        'all_trades': all_trades,
        'ticker_results': ticker_results,
        'cutoff_date': cutoff_date,
        'test_end_date': test_end_date,
    }


def print_results(results: Dict) -> None:
    """Print detailed results."""

    all_trades = results['all_trades']
    ticker_results = results['ticker_results']
    cutoff_date = results['cutoff_date']
    test_end_date = results['test_end_date']

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print(f"Test Period: {cutoff_date.strftime('%Y-%m-%d')} ~ {test_end_date.strftime('%Y-%m-%d')}")
    print("=" * 70)

    if not all_trades:
        print("\nNo trades generated!")
        return

    trades_df = pd.DataFrame(all_trades)

    total_trades = len(trades_df)
    wins = (trades_df.net_pnl > 0).sum()
    losses = total_trades - wins
    win_rate = wins / total_trades * 100

    total_pnl = trades_df.net_pnl.sum()
    avg_pnl = trades_df.net_pnl.mean()
    avg_win = trades_df[trades_df.net_pnl > 0].net_pnl.mean() if wins > 0 else 0
    avg_loss = trades_df[trades_df.net_pnl <= 0].net_pnl.mean() if losses > 0 else 0

    target_hits = (trades_df.exit_reason == 'target_hit').sum()
    stop_losses = (trades_df.exit_reason == 'stop_loss').sum()
    time_limits = (trades_df.exit_reason == 'time_limit').sum()

    print(f"\n=== Trade Statistics ===")
    print(f"Total trades:     {total_trades:,}")
    print(f"Wins:             {wins:,} ({win_rate:.1f}%)")
    print(f"Losses:           {losses:,} ({100-win_rate:.1f}%)")
    print(f"Target hit:       {target_hits} ({target_hits/total_trades*100:.1f}%)")
    print(f"Stop loss:        {stop_losses} ({stop_losses/total_trades*100:.1f}%)")
    print(f"Time limit:       {time_limits} ({time_limits/total_trades*100:.1f}%)")

    print(f"\n=== Profit Analysis ===")
    print(f"Total return:     {total_pnl:+.2f}%")
    print(f"Avg per trade:    {avg_pnl:+.4f}%")
    print(f"Avg win:          {avg_win:+.4f}%")
    print(f"Avg loss:         {avg_loss:+.4f}%")

    # Profit factor
    total_profit = trades_df[trades_df.net_pnl > 0].net_pnl.sum()
    total_loss_amt = abs(trades_df[trades_df.net_pnl <= 0].net_pnl.sum())
    profit_factor = total_profit / total_loss_amt if total_loss_amt > 0 else float('inf')
    print(f"Profit Factor:    {profit_factor:.2f}")

    # Daily breakdown
    trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
    daily_stats = trades_df.groupby('date').agg({
        'net_pnl': ['sum', 'count', lambda x: (x > 0).sum() / len(x) * 100]
    }).reset_index()
    daily_stats.columns = ['date', 'pnl', 'trades', 'win_rate']

    print(f"\n=== Daily Breakdown ===")
    print(f"{'Date':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<12}")
    print("-" * 44)
    for _, row in daily_stats.iterrows():
        print(f"{row['date']}  {int(row['trades']):<8} {row['win_rate']:.1f}%     {row['pnl']:+.2f}%")

    # Per ticker breakdown (top 10)
    print(f"\n=== Top 10 Tickers by Win Rate ===")
    ticker_stats = sorted(ticker_results, key=lambda x: x['win_rate'], reverse=True)
    print(f"{'Ticker':<8} {'Trades':<8} {'Win Rate':<10} {'Avg P&L':<10} {'Total P&L':<12}")
    print("-" * 52)
    for ts in ticker_stats[:10]:
        print(f"{ts['ticker']:<8} {ts['trades']:<8} {ts['win_rate']:.1f}%     {ts['avg_pnl']:+.3f}%   {ts['total_pnl']:+.2f}%")

    # A-grade tickers (Win Rate >= 60%, Trades >= 3)
    a_grade = [ts for ts in ticker_stats if ts['win_rate'] >= 60 and ts['trades'] >= 3]

    print(f"\n=== A-Grade Tickers (WR >= 60%, Trades >= 3) ===")
    print(f"Count: {len(a_grade)} tickers")

    if a_grade:
        print(f"\n{'Ticker':<8} {'Trades':<8} {'Win Rate':<10} {'Total P&L':<12}")
        print("-" * 40)
        for ts in a_grade:
            print(f"{ts['ticker']:<8} {ts['trades']:<8} {ts['win_rate']:.1f}%     {ts['total_pnl']:+.2f}%")

        # A-grade only stats
        a_grade_trades = [t for t in all_trades if t['ticker'] in [ts['ticker'] for ts in a_grade]]
        if a_grade_trades:
            a_wins = sum(1 for t in a_grade_trades if t['net_pnl'] > 0)
            a_total = len(a_grade_trades)
            a_pnl = sum(t['net_pnl'] for t in a_grade_trades)
            print(f"\n--- A-Grade Only Performance ---")
            print(f"Trades: {a_total}")
            print(f"Win Rate: {a_wins/a_total*100:.1f}%")
            print(f"Total P&L: {a_pnl:+.2f}%")

    # Expected returns calculation
    test_days = (test_end_date - cutoff_date).days
    if test_days > 0:
        daily_return = total_pnl / test_days
        annual_return = daily_return * 252

        print(f"\n=== Annualized Projection ===")
        print(f"Test period:        {test_days} days")
        print(f"Daily return:       {daily_return:+.4f}%")
        print(f"Annual return:      {annual_return:+.1f}%")

        initial_capital = 10000
        expected_annual = initial_capital * (annual_return / 100)
        print(f"\n=== Expected Returns ($10,000 capital) ===")
        print(f"Annual profit:      ${expected_annual:+,.2f}")
        print(f"Monthly profit:     ${expected_annual/12:+,.2f}")


def save_results(results: Dict, output_path: Path) -> None:
    """Save results to JSON file."""

    # Convert datetime objects to strings
    serializable_trades = []
    for trade in results['all_trades']:
        t = trade.copy()
        t['entry_time'] = t['entry_time'].isoformat() if isinstance(t['entry_time'], datetime) else str(t['entry_time'])
        serializable_trades.append(t)

    output = {
        'cutoff_date': results['cutoff_date'].strftime('%Y-%m-%d'),
        'test_end_date': results['test_end_date'].strftime('%Y-%m-%d'),
        'total_trades': len(results['all_trades']),
        'ticker_results': results['ticker_results'],
        'trades': serializable_trades,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="December 10 Backtest Simulation")
    parser.add_argument('--cutoff-date', type=str, default='2025-12-10',
                        help='Training cutoff date (YYYY-MM-DD)')
    parser.add_argument('--test-days', type=int, default=10,
                        help='Number of test days after cutoff')
    parser.add_argument('--model-type', choices=['xgboost', 'lightgbm'], default='lightgbm')
    parser.add_argument('--max-tickers', type=int, help='Maximum tickers to process')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Parse dates
    cutoff_date = datetime.strptime(args.cutoff_date, '%Y-%m-%d')
    test_end_date = cutoff_date + timedelta(days=args.test_days)

    # Run backtest
    results = run_backtest(
        cutoff_date=cutoff_date,
        test_end_date=test_end_date,
        model_type=args.model_type,
        max_tickers=args.max_tickers,
        verbose=args.verbose
    )

    # Print results
    print_results(results)

    # Save results
    output_path = project_root / "data" / f"backtest_{args.cutoff_date.replace('-', '')}.json"
    save_results(results, output_path)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
