"""
Backtest Simulation Script
Calculates expected returns from simulated trading

Features:
- Immediate re-entry after trade exit (maximizes capital utilization)
- No fixed interval between trades
- Sequential simulation with continuous position management
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import numpy as np
from config.settings import settings

def run_simulation():
    print('=== Backtest Simulation (Immediate Re-entry) ===\n')

    # Database connection
    db_path = settings.DATABASE_URL.replace('sqlite:///', '')
    conn = sqlite3.connect(db_path)

    # Get minute bars data
    print('Loading data...')
    bars_df = pd.read_sql_query('''
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM minute_bars
        ORDER BY timestamp
    ''', conn)
    bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])

    print(f'Total minute bars: {len(bars_df):,}')
    print(f'Period: {bars_df.timestamp.min().strftime("%Y-%m-%d")} to {bars_df.timestamp.max().strftime("%Y-%m-%d")}')
    days = (bars_df.timestamp.max() - bars_df.timestamp.min()).days
    print(f'Data period: {days} days')
    print(f'Tickers: {bars_df.symbol.nunique()}')

    # Simulation parameters
    THRESHOLD = 0.70  # 70% probability threshold
    TARGET_PERCENT = 1.0  # 1% target
    STOP_LOSS_PERCENT = 2.0  # 2% stop loss
    HORIZON_BARS = 12  # 12 bars = 60 minutes with 5-min bars
    COMMISSION = 0.2  # 0.2% round-trip commission
    POSITION_SIZE = 10000  # $10,000 per trade

    print(f'\n=== Simulation Settings ===')
    print(f'Entry threshold: {THRESHOLD:.0%}')
    print(f'Target profit: {TARGET_PERCENT}%')
    print(f'Stop loss: {STOP_LOSS_PERCENT}%')
    print(f'Max hold time: {HORIZON_BARS} bars (60 min with 5-min bars)')
    print(f'Commission (round-trip): {COMMISSION}%')
    print(f'Position size: ${POSITION_SIZE:,}')
    print(f'Re-entry mode: IMMEDIATE (after each trade exit)')

    # Get unique tickers
    tickers = bars_df.symbol.unique()
    print(f'\nAnalyzing {len(tickers)} tickers...')

    # Run backtest simulation with immediate re-entry
    all_trades = []

    for ticker in tickers:
        ticker_bars = bars_df[bars_df.symbol == ticker].copy().reset_index(drop=True)
        if len(ticker_bars) < 200:
            continue

        # Calculate momentum indicators
        ticker_bars['returns_5'] = ticker_bars['close'].pct_change(5)
        ticker_bars['returns_15'] = ticker_bars['close'].pct_change(15)
        ticker_bars['vol_ratio'] = ticker_bars['volume'] / ticker_bars['volume'].rolling(20).mean()

        # IMMEDIATE RE-ENTRY: Sequential simulation
        current_idx = 50  # Start after warmup period
        total_bars = len(ticker_bars)

        while current_idx < total_bars - HORIZON_BARS:
            try:
                row = ticker_bars.iloc[current_idx]

                if pd.isna(row['returns_5']) or pd.isna(row['vol_ratio']):
                    current_idx += 1
                    continue

                # Calculate pseudo-probability based on momentum
                momentum_score = (row['returns_5'] * 100 + row['returns_15'] * 50) / 2
                vol_boost = min(row['vol_ratio'], 2.0) - 1.0

                base_prob = 0.5 + (momentum_score / 10) + (vol_boost * 0.05)
                prob = max(0.4, min(0.9, base_prob))

                if prob < THRESHOLD:
                    current_idx += 1
                    continue

                # Entry signal found
                entry_price = row['close']
                entry_time = row['timestamp']

                # Simulate trade
                exit_idx = None
                exit_price = None
                exit_reason = None

                for j in range(current_idx + 1, min(current_idx + HORIZON_BARS + 1, total_bars)):
                    future_row = ticker_bars.iloc[j]
                    high_return = (future_row['high'] - entry_price) / entry_price * 100
                    low_return = (future_row['low'] - entry_price) / entry_price * 100

                    if high_return >= TARGET_PERCENT:
                        exit_price = entry_price * (1 + TARGET_PERCENT / 100)
                        exit_reason = 'target_hit'
                        exit_idx = j
                        break

                    if low_return <= -STOP_LOSS_PERCENT:
                        exit_price = entry_price * (1 - STOP_LOSS_PERCENT / 100)
                        exit_reason = 'stop_loss'
                        exit_idx = j
                        break

                if exit_price is None:
                    exit_idx = min(current_idx + HORIZON_BARS, total_bars - 1)
                    exit_price = ticker_bars.iloc[exit_idx]['close']
                    exit_reason = 'time_limit'

                gross_pnl = (exit_price - entry_price) / entry_price * 100
                net_pnl = gross_pnl - COMMISSION
                dollar_pnl = POSITION_SIZE * (net_pnl / 100)

                all_trades.append({
                    'ticker': ticker,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'probability': prob,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'dollar_pnl': dollar_pnl
                })

                # IMMEDIATE RE-ENTRY: Move to bar right after exit
                current_idx = exit_idx + 1

            except Exception:
                current_idx += 1
                continue

    print(f'\n=== Backtest Results ===')
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        total_trades = len(trades_df)
        wins = (trades_df.net_pnl > 0).sum()
        losses = total_trades - wins
        win_rate = wins / total_trades * 100

        total_pnl_pct = trades_df.net_pnl.sum()
        total_pnl_dollar = trades_df.dollar_pnl.sum()
        avg_pnl = trades_df.net_pnl.mean()
        avg_win = trades_df[trades_df.net_pnl > 0].net_pnl.mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df.net_pnl <= 0].net_pnl.mean() if losses > 0 else 0

        target_hits = (trades_df.exit_reason == 'target_hit').sum()
        time_limits = (trades_df.exit_reason == 'time_limit').sum()
        stop_losses = (trades_df.exit_reason == 'stop_loss').sum()

        print(f'Total trades: {total_trades:,}')
        print(f'Wins: {wins:,} ({win_rate:.1f}%)')
        print(f'Losses: {losses:,}')
        print(f'Target hit: {target_hits} ({target_hits/total_trades*100:.1f}%)')
        print(f'Stop loss: {stop_losses} ({stop_losses/total_trades*100:.1f}%)')
        print(f'Time limit: {time_limits} ({time_limits/total_trades*100:.1f}%)')

        print(f'\n=== Profit Analysis ===')
        print(f'Total return: {total_pnl_pct:.2f}%')
        print(f'Total P&L: ${total_pnl_dollar:,.2f}')
        print(f'Avg return per trade: {avg_pnl:.3f}%')
        print(f'Avg win: +{avg_win:.3f}%')
        print(f'Avg loss: {avg_loss:.3f}%')

        # Profit factor
        total_loss = abs(trades_df[trades_df.net_pnl <= 0].net_pnl.sum())
        if total_loss > 0:
            profit_factor = trades_df[trades_df.net_pnl > 0].net_pnl.sum() / total_loss
            print(f'Profit Factor: {profit_factor:.2f}')

        # Calculate annualized return
        trade_days = (trades_df.entry_time.max() - trades_df.entry_time.min()).days
        if trade_days > 0:
            daily_pnl = total_pnl_pct / trade_days
            annual_pnl = daily_pnl * 252

            print(f'\n=== Annualized Return Estimate ===')
            print(f'Backtest period: {trade_days} days')
            print(f'Daily return: {daily_pnl:.4f}%')
            print(f'Annualized return: {annual_pnl:.1f}%')

            initial_capital = 10000
            expected_annual_return = initial_capital * (annual_pnl / 100)

            print(f'\n=== Expected Annual Returns ($10,000 capital) ===')
            print(f'Expected annual profit: ${expected_annual_return:,.2f}')
            print(f'Expected final capital: ${initial_capital + expected_annual_return:,.2f}')
            print(f'Expected monthly profit: ${expected_annual_return/12:,.2f}')

            # Risk-adjusted metrics
            if len(trades_df) > 1:
                returns_std = trades_df.net_pnl.std()
                sharpe = (avg_pnl / returns_std) * np.sqrt(252 * 6) if returns_std > 0 else 0  # ~6 trades per day
                print(f'\nSharpe Ratio (approx): {sharpe:.2f}')

                # Max drawdown calculation
                cumulative = trades_df.net_pnl.cumsum()
                rolling_max = cumulative.cummax()
                drawdown = cumulative - rolling_max
                max_drawdown = drawdown.min()
                print(f'Max Drawdown: {max_drawdown:.2f}%')
    else:
        print('No trades generated.')

    conn.close()

if __name__ == '__main__':
    run_simulation()
