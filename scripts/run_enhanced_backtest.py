#!/usr/bin/env python
"""
Enhanced Backtest with All Improvements

Implements all 4 CEO-requested improvements:
1. S+ Grade: win_rate >= 80% AND profit_factor >= 4.0
2. Rebalancing: Every 5 days recalculation
3. Transaction Costs: commission 0.25%, slippage 0.1%, stop loss -5%
4. Concentration Strategy: S+ -> top 2 stocks (50%), else A-grade 4 stocks (25%)

Usage:
    python run_enhanced_backtest.py --cutoff-date 2025-12-10 --test-days 10
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester.investment_strategy import (
    InvestmentStrategy,
    StockGrade,
    StockPerformance,
    TransactionCosts,
    calculate_enhanced_metrics
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level
    )


def get_database_path() -> Path:
    """Get database path."""
    db_path = PROJECT_ROOT / "data" / "predictions.db"
    if not db_path.exists():
        db_path = PROJECT_ROOT / "predictions.db"
    return db_path


def load_model_predictions(
    db_path: Path,
    cutoff_date: str,
    test_days: int
) -> pd.DataFrame:
    """Load model predictions from database."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        timestamp,
        ticker,
        model_type,
        up_prob,
        actual_outcome,
        profit_pct
    FROM predictions
    WHERE date(timestamp) >= date(?)
    AND date(timestamp) <= date(?, '+' || ? || ' days')
    ORDER BY timestamp
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=[cutoff_date, cutoff_date, test_days]
    )
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} predictions from {cutoff_date} +{test_days} days")

    return df


def calculate_ticker_performance(
    predictions: pd.DataFrame,
    costs: TransactionCosts
) -> List[Dict]:
    """
    Calculate performance metrics for each ticker.

    Returns list of dicts with ticker performance data.
    """
    performance_data = []

    for ticker in predictions['ticker'].unique():
        ticker_data = predictions[predictions['ticker'] == ticker]

        # Calculate trades (where we would have entered)
        trades = ticker_data[ticker_data['up_prob'] >= 0.60].copy()

        if len(trades) == 0:
            continue

        # Calculate win rate
        wins = trades[trades['actual_outcome'] == 1]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0

        # Calculate profit factor
        profits = trades[trades['profit_pct'] > 0]['profit_pct'].sum()
        losses = abs(trades[trades['profit_pct'] < 0]['profit_pct'].sum())
        profit_factor = profits / losses if losses > 0 else 999.99

        # Calculate total return (with transaction costs)
        total_return = trades['profit_pct'].sum() - (len(trades) * costs.round_trip_cost)

        performance_data.append({
            'ticker': ticker,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_trades': len(trades)
        })

    return performance_data


def run_period_simulation(
    predictions: pd.DataFrame,
    strategy: InvestmentStrategy,
    start_date: datetime,
    end_date: datetime,
    rebalance_days: int = 5,
    initial_capital: float = 100000.0
) -> Dict:
    """
    Run simulation for a period with rebalancing.

    Args:
        predictions: DataFrame with predictions
        strategy: InvestmentStrategy instance
        start_date: Period start
        end_date: Period end
        rebalance_days: Days between rebalancing
        initial_capital: Starting capital

    Returns:
        Simulation results dict
    """
    results = {
        'periods': [],
        'trades': [],
        'total_return': 0.0,
        'final_capital': initial_capital
    }

    current_date = start_date
    capital = initial_capital
    period_num = 0

    while current_date < end_date:
        period_end = min(current_date + timedelta(days=rebalance_days), end_date)

        # Get predictions for this period
        period_preds = predictions[
            (predictions['timestamp'].dt.date >= current_date.date()) &
            (predictions['timestamp'].dt.date < period_end.date())
        ]

        if len(period_preds) == 0:
            current_date = period_end
            continue

        # Calculate performance for previous period (for grading)
        if period_num == 0:
            # Use training data performance for first period
            prev_preds = predictions[predictions['timestamp'].dt.date < current_date.date()]
        else:
            # Use last 5 days performance
            lookback_start = current_date - timedelta(days=rebalance_days)
            prev_preds = predictions[
                (predictions['timestamp'].dt.date >= lookback_start.date()) &
                (predictions['timestamp'].dt.date < current_date.date())
            ]

        # Get ticker performance
        perf_data = calculate_ticker_performance(prev_preds, strategy.costs)

        # Evaluate stocks and get allocation
        performances = strategy.evaluate_stocks(perf_data)
        allocation = strategy.calculate_allocation(performances)

        # Simulate trades for this period using allocation
        period_return = 0.0
        period_trades = []

        for ticker, weight in allocation.allocations.items():
            ticker_preds = period_preds[period_preds['ticker'] == ticker]
            if len(ticker_preds) == 0:
                continue

            # Simulate trades for this ticker
            for _, pred in ticker_preds.iterrows():
                if pred['up_prob'] >= 0.60:  # Entry threshold
                    # Check stop loss
                    profit = pred['profit_pct']
                    stopped, adj_return = strategy.apply_stop_loss(profit)

                    # Apply transaction costs
                    net_return = adj_return - strategy.costs.round_trip_cost

                    # Weight by allocation
                    weighted_return = net_return * weight

                    period_return += weighted_return
                    period_trades.append({
                        'ticker': ticker,
                        'timestamp': pred['timestamp'],
                        'weight': weight,
                        'gross_return': profit,
                        'net_return': net_return,
                        'weighted_return': weighted_return,
                        'stopped': stopped
                    })

        # Update capital
        capital_change = capital * (period_return / 100)
        capital += capital_change

        # Record period results
        period_result = {
            'period': period_num,
            'start': current_date.strftime('%Y-%m-%d'),
            'end': period_end.strftime('%Y-%m-%d'),
            'allocation': allocation.allocations,
            'strategy': allocation.strategy,
            'num_trades': len(period_trades),
            'period_return': period_return,
            'capital_start': capital - capital_change,
            'capital_end': capital,
            's_plus_stocks': [p.ticker for p in performances if p.grade == StockGrade.S_PLUS],
            'a_grade_stocks': [p.ticker for p in performances if p.grade == StockGrade.A]
        }

        results['periods'].append(period_result)
        results['trades'].extend(period_trades)

        logger.info(
            f"Period {period_num} ({current_date.strftime('%m/%d')} - {period_end.strftime('%m/%d')}): "
            f"{allocation.strategy} strategy, {len(period_trades)} trades, "
            f"return: {period_return:+.2f}%, capital: ${capital:,.0f}"
        )

        current_date = period_end
        period_num += 1

    results['final_capital'] = capital
    results['total_return'] = ((capital - initial_capital) / initial_capital) * 100

    return results


def print_results(results: Dict, strategy: InvestmentStrategy):
    """Print simulation results."""
    print("\n" + "=" * 60)
    print("ENHANCED BACKTEST RESULTS")
    print("=" * 60)

    # Configuration
    print("\n[Configuration]")
    print(f"  Rebalance Period: Every {strategy.rebalance_days} days")
    print(f"  Commission: {strategy.costs.commission_pct}%")
    print(f"  Slippage: {strategy.costs.slippage_pct}%")
    print(f"  Stop Loss: -{strategy.costs.stop_loss_pct}%")

    # Overall Results
    print("\n[Overall Results]")
    print(f"  Initial Capital: $100,000")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return']:+.2f}%")
    print(f"  Total Trades: {len(results['trades'])}")

    # Period Summary
    print("\n[Period Summary]")
    for period in results['periods']:
        s_plus = period.get('s_plus_stocks', [])
        a_grade = period.get('a_grade_stocks', [])

        print(f"\n  Period {period['period']}: {period['start']} ~ {period['end']}")
        print(f"    Strategy: {period['strategy']}")
        print(f"    S+ Stocks: {', '.join(s_plus) if s_plus else 'None'}")
        print(f"    A-Grade: {', '.join(a_grade) if a_grade else 'None'}")
        print(f"    Allocation: {period['allocation']}")
        print(f"    Trades: {period['num_trades']}, Return: {period['period_return']:+.2f}%")

    # Trade Statistics
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])

        print("\n[Trade Statistics]")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Avg Return per Trade: {trades_df['net_return'].mean():+.2f}%")
        print(f"  Stop Loss Triggered: {trades_df['stopped'].sum()} times")

        # By ticker
        print("\n[By Ticker]")
        ticker_stats = trades_df.groupby('ticker').agg({
            'weighted_return': 'sum',
            'stopped': 'sum'
        }).sort_values('weighted_return', ascending=False)

        for ticker, row in ticker_stats.iterrows():
            print(f"    {ticker}: {row['weighted_return']:+.2f}% (stops: {int(row['stopped'])})")

    print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced Backtest with All Improvements')
    parser.add_argument('--cutoff-date', default='2025-12-10',
                        help='Training cutoff date (default: 2025-12-10)')
    parser.add_argument('--test-days', type=int, default=10,
                        help='Number of test days (default: 10)')
    parser.add_argument('--rebalance-days', type=int, default=5,
                        help='Rebalancing period in days (default: 5)')
    parser.add_argument('--commission', type=float, default=0.25,
                        help='Commission percentage (default: 0.25)')
    parser.add_argument('--slippage', type=float, default=0.10,
                        help='Slippage percentage (default: 0.10)')
    parser.add_argument('--stop-loss', type=float, default=5.0,
                        help='Stop loss percentage (default: 5.0)')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path for results (JSON)')

    args = parser.parse_args()
    setup_logging(args.verbose)

    print("\n" + "=" * 60)
    print(" FiveForFree Enhanced Backtest")
    print(" All 4 CEO Improvements Implemented")
    print("=" * 60)

    # Setup
    db_path = get_database_path()
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Running in simulation mode with sample data...")

        # Create sample performance data for demonstration
        sample_data = [
            {'ticker': 'NXPI', 'win_rate': 0.857, 'profit_factor': 4.35, 'total_return': 3.44, 'total_trades': 7},
            {'ticker': 'EBAY', 'win_rate': 0.833, 'profit_factor': 6.21, 'total_return': 2.24, 'total_trades': 6},
            {'ticker': 'LULU', 'win_rate': 0.789, 'profit_factor': 1.61, 'total_return': 3.60, 'total_trades': 19},
            {'ticker': 'NVDA', 'win_rate': 0.727, 'profit_factor': 4.31, 'total_return': 2.78, 'total_trades': 11},
            {'ticker': 'AAPL', 'win_rate': 0.650, 'profit_factor': 1.85, 'total_return': 1.50, 'total_trades': 20},
            {'ticker': 'MSFT', 'win_rate': 0.600, 'profit_factor': 1.50, 'total_return': 0.80, 'total_trades': 15},
        ]

        # Create strategy
        costs = TransactionCosts(
            commission_pct=args.commission,
            slippage_pct=args.slippage,
            stop_loss_pct=args.stop_loss
        )
        strategy = InvestmentStrategy(
            rebalance_days=args.rebalance_days,
            transaction_costs=costs
        )

        # Evaluate stocks
        performances = strategy.evaluate_stocks(sample_data)

        print("\n[Stock Grades]")
        for perf in performances:
            print(f"  {perf.ticker}: {perf.grade.value} "
                  f"(WR: {perf.win_rate*100:.1f}%, PF: {perf.profit_factor:.2f})")

        # Get allocation
        allocation = strategy.calculate_allocation(performances)

        print(f"\n[Allocation Strategy: {allocation.strategy}]")
        for ticker, weight in allocation.allocations.items():
            print(f"  {ticker}: {weight*100:.1f}%")

        # Demonstrate S+ criteria
        print("\n[S+ Grade Criteria]")
        print("  - Win Rate >= 80%")
        print("  - Profit Factor >= 4.0")
        print(f"\n  S+ Stocks: {[p.ticker for p in performances if p.grade == StockGrade.S_PLUS]}")

        return

    # Load predictions
    cutoff = datetime.strptime(args.cutoff_date, '%Y-%m-%d')
    predictions = load_model_predictions(db_path, args.cutoff_date, args.test_days)

    if predictions.empty:
        logger.error("No predictions found for the specified period")
        return

    # Create strategy with custom costs
    costs = TransactionCosts(
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        stop_loss_pct=args.stop_loss
    )
    strategy = InvestmentStrategy(
        rebalance_days=args.rebalance_days,
        transaction_costs=costs
    )

    # Run simulation
    start_date = cutoff
    end_date = cutoff + timedelta(days=args.test_days)

    results = run_period_simulation(
        predictions=predictions,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        rebalance_days=args.rebalance_days,
        initial_capital=args.capital
    )

    # Print results
    print_results(results, strategy)

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            # Convert datetime objects to strings
            serializable = results.copy()
            for trade in serializable['trades']:
                if 'timestamp' in trade:
                    trade['timestamp'] = str(trade['timestamp'])
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
