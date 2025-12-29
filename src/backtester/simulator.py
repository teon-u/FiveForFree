"""
Backtest Simulator for NASDAQ Prediction System

Simulates trading based on model predictions using realistic rules:
- Long Only strategy
- Entry: up_prob >= threshold (default 60%)
- Exit: Target profit OR Stop Loss OR 60 minutes elapsed
- Commission: 0.2% round-trip
- Slippage: 0.05% per trade

Tracks individual trades and generates comprehensive results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
import pytz

warnings.filterwarnings('ignore')


def is_market_hours(timestamp: datetime) -> bool:
    """
    Check if given timestamp is within US market hours (9:30 AM - 4:00 PM ET).

    Args:
        timestamp: Datetime to check (can be naive or timezone-aware)

    Returns:
        True if within market hours, False otherwise
    """
    # Convert to Eastern Time if timezone-aware, or assume naive is ET
    eastern = pytz.timezone('US/Eastern')

    if timestamp.tzinfo is None:
        # Naive datetime - assume it's already in Eastern Time
        et_time = timestamp
    else:
        # Convert to Eastern Time
        et_time = timestamp.astimezone(eastern)

    # Check weekday (Monday=0, Friday=4)
    if et_time.weekday() > 4:
        return False

    # Check time range
    market_open = et_time.replace(
        hour=settings.MARKET_OPEN_HOUR,
        minute=settings.MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    market_close = et_time.replace(
        hour=settings.MARKET_CLOSE_HOUR,
        minute=settings.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )

    return market_open <= et_time < market_close


@dataclass
class Trade:
    """
    Represents a single trade in the backtest.

    Attributes:
        ticker: Stock ticker symbol
        entry_time: When the trade was entered
        entry_price: Entry price
        exit_time: When the trade was exited
        exit_price: Exit price
        exit_reason: 'target_hit', 'stop_loss', or 'time_limit'
        probability: Model's predicted probability at entry
        profit_pct: Net profit percentage (after commission and slippage)
        duration_minutes: Trade duration in minutes
        model_type: Type of model used for prediction
        target: 'up' or 'down'
        position_size: Position size in dollars
        shares: Number of shares traded
    """
    ticker: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    probability: float
    profit_pct: float
    duration_minutes: float
    model_type: Optional[str] = None
    target: str = 'up'
    position_size: float = 1000.0
    shares: int = 0

    @property
    def is_win(self) -> bool:
        """Check if trade was profitable."""
        return self.profit_pct > 0

    @property
    def profit_dollars(self) -> float:
        """Calculate dollar profit based on actual position size."""
        return self.position_size * (self.profit_pct / 100)

    def to_dict(self) -> dict:
        """Convert trade to dictionary with native Python types."""
        return {
            'ticker': str(self.ticker),
            'entry_time': self.entry_time,
            'entry_price': float(self.entry_price),
            'exit_time': self.exit_time,
            'exit_price': float(self.exit_price),
            'exit_reason': str(self.exit_reason),
            'probability': float(self.probability),
            'profit_pct': float(self.profit_pct),
            'duration_minutes': float(self.duration_minutes),
            'model_type': str(self.model_type) if self.model_type else None,
            'target': str(self.target),
            'is_win': bool(self.is_win),
            'profit_dollars': float(self.profit_dollars),
            'position_size': float(self.position_size),
            'shares': int(self.shares)
        }


@dataclass
class BacktestResult:
    """
    Comprehensive backtest results.

    Attributes:
        ticker: Stock ticker
        model_type: Model type used
        target: 'up' or 'down'
        trades: List of all trades
        start_time: Backtest start time
        end_time: Backtest end time
        total_predictions: Total predictions made
        total_trades: Number of trades taken
        metadata: Additional metadata
    """
    ticker: str
    model_type: str
    target: str
    trades: List[Trade] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_predictions: int = 0
    total_trades: int = 0
    metadata: dict = field(default_factory=dict)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the results."""
        self.trades.append(trade)
        self.total_trades += 1

    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([trade.to_dict() for trade in self.trades])

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.trades:
            return {
                'ticker': self.ticker,
                'model_type': self.model_type,
                'target': self.target,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'total_profit': 0.0
            }

        wins = [t for t in self.trades if t.is_win]
        losses = [t for t in self.trades if not t.is_win]

        return {
            'ticker': self.ticker,
            'model_type': self.model_type,
            'target': self.target,
            'total_predictions': self.total_predictions,
            'total_trades': self.total_trades,
            'trade_rate': self.total_trades / self.total_predictions if self.total_predictions > 0 else 0.0,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) if self.trades else 0.0,
            'avg_profit': np.mean([t.profit_pct for t in self.trades]),
            'avg_win': np.mean([t.profit_pct for t in wins]) if wins else 0.0,
            'avg_loss': np.mean([t.profit_pct for t in losses]) if losses else 0.0,
            'total_profit': sum(t.profit_pct for t in self.trades),
            'best_trade': max((t.profit_pct for t in self.trades), default=0.0),
            'worst_trade': min((t.profit_pct for t in self.trades), default=0.0),
            'avg_duration': np.mean([t.duration_minutes for t in self.trades]),
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class BacktestSimulator:
    """
    Backtest simulator for NASDAQ prediction models.

    Simulates Long Only trading with the following rules:
    - Entry: When up_prob >= probability_threshold
    - Exit: Target profit OR Stop Loss OR 60 minutes time limit
    - Commission: 0.2% round-trip (0.1% entry + 0.1% exit)
    - Slippage: 0.05% per trade (applied to both entry and exit)

    Tracks all trades and integrates with model accuracy tracking.
    """

    def __init__(
        self,
        probability_threshold: float = None,
        target_percent: float = None,
        prediction_horizon_minutes: int = None,
        commission_pct: float = None,
        stop_loss_percent: float = None,
        slippage_percent: float = None,
        market_hours_only: bool = False
    ):
        """
        Initialize backtest simulator.

        Args:
            probability_threshold: Minimum probability to enter trade (default from settings)
            target_percent: Target profit percentage (default from settings)
            prediction_horizon_minutes: Max time to hold position (default from settings)
            commission_pct: One-way commission percentage (default from settings)
            stop_loss_percent: Stop loss percentage (default from settings)
            slippage_percent: Slippage percentage per trade (default from settings)
            market_hours_only: If True, only trade during market hours (9:30-16:00 ET)
        """
        self.probability_threshold = probability_threshold or settings.PROBABILITY_THRESHOLD
        self.target_percent = target_percent or settings.TARGET_PERCENT
        self.prediction_horizon_minutes = prediction_horizon_minutes or settings.PREDICTION_HORIZON_MINUTES
        self.commission_pct = commission_pct or settings.COMMISSION_PERCENT
        self.commission_round_trip = self.commission_pct * 2
        self.stop_loss_percent = stop_loss_percent or getattr(settings, 'STOP_LOSS_PERCENT', 0.5)
        self.slippage_percent = slippage_percent or getattr(settings, 'SLIPPAGE_PERCENT', 0.05)
        self.slippage_round_trip = self.slippage_percent * 2
        self.market_hours_only = market_hours_only

        logger.info(
            f"BacktestSimulator initialized: threshold={self.probability_threshold:.1%}, "
            f"target={self.target_percent}%, stop_loss={self.stop_loss_percent}%, "
            f"horizon={self.prediction_horizon_minutes}min, "
            f"commission={self.commission_round_trip:.2%}, slippage={self.slippage_round_trip:.2%}, "
            f"market_hours_only={self.market_hours_only}"
        )

    def simulate_trade(
        self,
        ticker: str,
        entry_time: datetime,
        entry_price: float,
        minute_bars: pd.DataFrame,
        up_prob: float,
        model_type: str = None,
        target: str = 'up'
    ) -> Optional[Trade]:
        """
        Simulate a single trade with stop loss and slippage.

        Args:
            ticker: Stock ticker
            entry_time: Entry timestamp
            entry_price: Entry price (before slippage)
            minute_bars: DataFrame with minute-level price data
            up_prob: Predicted up probability
            model_type: Type of model used
            target: 'up' or 'down'

        Returns:
            Trade object if trade was taken, None if no trade
        """
        # Check if within market hours (9:30 AM - 4:00 PM ET)
        if not is_market_hours(entry_time):
            return None

        # Check if probability meets threshold
        if up_prob < self.probability_threshold:
            return None

        # Apply slippage to entry (worse entry = higher price for long)
        actual_entry_price = entry_price * (1 + self.slippage_percent / 100)

        # Get future bars within prediction horizon
        end_time = entry_time + timedelta(minutes=self.prediction_horizon_minutes)

        future_bars = minute_bars[
            (minute_bars['timestamp'] > entry_time) &
            (minute_bars['timestamp'] <= end_time)
        ].copy()

        if len(future_bars) == 0:
            logger.warning(f"No future bars available for {ticker} at {entry_time}")
            return None

        # Ensure bars are sorted by time
        future_bars = future_bars.sort_values('timestamp')

        # Simulate trade execution
        exit_price = None
        exit_time = None
        exit_reason = None

        # Iterate through each minute to find exit
        for idx, row in future_bars.iterrows():
            timestamp = row['timestamp']
            high = row['high']
            low = row['low']
            close = row['close']

            # Calculate intrabar returns from actual entry price
            high_return = ((high - actual_entry_price) / actual_entry_price) * 100
            low_return = ((low - actual_entry_price) / actual_entry_price) * 100

            # Check stop loss FIRST (assume stop loss hit before target in same bar)
            if low_return <= -self.stop_loss_percent:
                # Stop loss hit - exit at stop loss price
                exit_price = actual_entry_price * (1 - self.stop_loss_percent / 100)
                exit_time = timestamp
                exit_reason = 'stop_loss'
                break

            # Check if target was hit
            if high_return >= self.target_percent:
                # Target hit - exit at target price
                exit_price = actual_entry_price * (1 + self.target_percent / 100)
                exit_time = timestamp
                exit_reason = 'target_hit'
                break

            # Check if time limit reached
            elapsed_minutes = (timestamp - entry_time).total_seconds() / 60
            if elapsed_minutes >= self.prediction_horizon_minutes:
                # Time limit - exit at close price
                exit_price = close
                exit_time = timestamp
                exit_reason = 'time_limit'
                break

            # Check if market is about to close (force exit before 4:00 PM)
            if not is_market_hours(timestamp):
                # Market closed - exit at close price
                exit_price = close
                exit_time = timestamp
                exit_reason = 'market_close'
                break

        # If loop completed without exit, use last bar
        if exit_price is None:
            last_bar = future_bars.iloc[-1]
            exit_price = last_bar['close']
            exit_time = last_bar['timestamp']
            exit_reason = 'time_limit'

        # Apply slippage to exit (worse exit = lower price for sell)
        # Note: For stop_loss and target_hit, slippage may cause slightly worse execution
        if exit_reason == 'time_limit':
            actual_exit_price = exit_price * (1 - self.slippage_percent / 100)
        else:
            # For stop_loss/target_hit, slippage is already factored in the limit price
            actual_exit_price = exit_price * (1 - self.slippage_percent / 100)

        # Calculate profit percentage (net of commission and slippage)
        gross_return = ((actual_exit_price - actual_entry_price) / actual_entry_price) * 100
        net_return = gross_return - self.commission_round_trip

        # Calculate duration
        duration_minutes = (exit_time - entry_time).total_seconds() / 60

        # Create trade object
        trade = Trade(
            ticker=ticker,
            entry_time=entry_time,
            entry_price=actual_entry_price,
            exit_time=exit_time,
            exit_price=actual_exit_price,
            exit_reason=exit_reason,
            probability=up_prob,
            profit_pct=net_return,
            duration_minutes=duration_minutes,
            model_type=model_type,
            target=target
        )

        return trade

    def simulate_predictions(
        self,
        ticker: str,
        predictions_df: pd.DataFrame,
        minute_bars: pd.DataFrame,
        model_type: str = None,
        target: str = 'up'
    ) -> BacktestResult:
        """
        Simulate trading based on a series of predictions.

        Args:
            ticker: Stock ticker
            predictions_df: DataFrame with columns [timestamp, probability]
            minute_bars: DataFrame with minute-level price data [timestamp, open, high, low, close, volume]
            model_type: Type of model used
            target: 'up' or 'down'

        Returns:
            BacktestResult with all trades and statistics
        """
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(predictions_df['timestamp']):
            predictions_df = predictions_df.copy()
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])

        if not pd.api.types.is_datetime64_any_dtype(minute_bars['timestamp']):
            minute_bars = minute_bars.copy()
            minute_bars['timestamp'] = pd.to_datetime(minute_bars['timestamp'])

        # Sort by timestamp
        predictions_df = predictions_df.sort_values('timestamp').reset_index(drop=True)
        minute_bars = minute_bars.sort_values('timestamp').reset_index(drop=True)

        # Initialize result
        result = BacktestResult(
            ticker=ticker,
            model_type=model_type or 'unknown',
            target=target,
            start_time=predictions_df['timestamp'].min(),
            end_time=predictions_df['timestamp'].max(),
            total_predictions=len(predictions_df)
        )

        # Simulate each prediction
        for idx, pred_row in predictions_df.iterrows():
            entry_time = pred_row['timestamp']
            probability = pred_row['probability']

            # Get entry price (close price at prediction time)
            entry_bars = minute_bars[minute_bars['timestamp'] == entry_time]
            if len(entry_bars) == 0:
                # Try to find closest bar
                closest_idx = (minute_bars['timestamp'] - entry_time).abs().argmin()
                entry_price = minute_bars.loc[closest_idx, 'close']
            else:
                entry_price = entry_bars.iloc[0]['close']

            # Simulate trade
            trade = self.simulate_trade(
                ticker=ticker,
                entry_time=entry_time,
                entry_price=entry_price,
                minute_bars=minute_bars,
                up_prob=probability,
                model_type=model_type,
                target=target
            )

            if trade is not None:
                result.add_trade(trade)

        logger.info(
            f"Backtest completed for {ticker} ({model_type}, {target}): "
            f"{result.total_trades} trades from {result.total_predictions} predictions"
        )

        return result

    def simulate_vectorized(
        self,
        ticker: str,
        minute_bars: pd.DataFrame,
        probabilities: np.ndarray,
        model_type: str = None,
        target: str = 'up'
    ) -> BacktestResult:
        """
        Vectorized backtest simulation for better performance.

        Uses numpy operations where possible to speed up computation.

        Args:
            ticker: Stock ticker
            minute_bars: DataFrame with minute-level data
            probabilities: Array of probabilities aligned with minute_bars
            model_type: Type of model
            target: 'up' or 'down'

        Returns:
            BacktestResult with all trades
        """
        if len(minute_bars) != len(probabilities):
            raise ValueError("minute_bars and probabilities must have same length")

        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(minute_bars['timestamp']):
            minute_bars = minute_bars.copy()
            minute_bars['timestamp'] = pd.to_datetime(minute_bars['timestamp'])

        # Sort by timestamp
        minute_bars = minute_bars.sort_values('timestamp').reset_index(drop=True)

        # Initialize result
        result = BacktestResult(
            ticker=ticker,
            model_type=model_type or 'unknown',
            target=target,
            start_time=minute_bars['timestamp'].min(),
            end_time=minute_bars['timestamp'].max(),
            total_predictions=len(minute_bars)
        )

        # Filter to entries that meet probability threshold
        entry_mask = probabilities >= self.probability_threshold
        entry_indices = np.where(entry_mask)[0]

        logger.info(f"Found {len(entry_indices)} potential entries from {len(minute_bars)} bars")

        # Simulate each trade
        for entry_idx in entry_indices:
            # Skip if not enough future data
            if entry_idx + self.prediction_horizon_minutes >= len(minute_bars):
                continue

            entry_time = minute_bars.loc[entry_idx, 'timestamp']
            entry_price = minute_bars.loc[entry_idx, 'close']
            probability = probabilities[entry_idx]

            # Get future bars
            future_start = entry_idx + 1
            future_end = min(entry_idx + self.prediction_horizon_minutes + 1, len(minute_bars))
            future_bars = minute_bars.iloc[future_start:future_end]

            # Simulate trade
            trade = self.simulate_trade(
                ticker=ticker,
                entry_time=entry_time,
                entry_price=entry_price,
                minute_bars=minute_bars,
                up_prob=probability,
                model_type=model_type,
                target=target
            )

            if trade is not None:
                result.add_trade(trade)

        logger.info(
            f"Vectorized backtest completed: {result.total_trades} trades "
            f"from {len(entry_indices)} signals"
        )

        return result

    def simulate_with_reentry(
        self,
        ticker: str,
        minute_bars: pd.DataFrame,
        probabilities: np.ndarray,
        model_type: str = None,
        target: str = 'up'
    ) -> BacktestResult:
        """
        Backtest simulation with immediate re-entry after trade exit.

        After a trade exits (target hit, stop loss, or time limit), immediately check
        the next bar for a new entry signal. This maximizes capital utilization.

        Args:
            ticker: Stock ticker
            minute_bars: DataFrame with minute-level data
            probabilities: Array of probabilities aligned with minute_bars
            model_type: Type of model
            target: 'up' or 'down'

        Returns:
            BacktestResult with all trades
        """
        if len(minute_bars) != len(probabilities):
            raise ValueError("minute_bars and probabilities must have same length")

        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(minute_bars['timestamp']):
            minute_bars = minute_bars.copy()
            minute_bars['timestamp'] = pd.to_datetime(minute_bars['timestamp'])

        # Sort by timestamp and reset index
        minute_bars = minute_bars.sort_values('timestamp').reset_index(drop=True)

        # Initialize result
        result = BacktestResult(
            ticker=ticker,
            model_type=model_type or 'unknown',
            target=target,
            start_time=minute_bars['timestamp'].min(),
            end_time=minute_bars['timestamp'].max(),
            total_predictions=len(minute_bars)
        )

        # Sequential simulation with immediate re-entry
        current_idx = 0
        total_bars = len(minute_bars)
        signals_checked = 0

        # Drawdown tracking
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        max_drawdown_limit = getattr(settings, 'MAX_DRAWDOWN_PERCENT', 20.0)
        trading_halted = False

        while current_idx < total_bars - self.prediction_horizon_minutes:
            signals_checked += 1
            probability = probabilities[current_idx]
            entry_time = minute_bars.loc[current_idx, 'timestamp']

            # Check if within market hours (9:30 AM - 4:00 PM ET) - only if enabled
            if self.market_hours_only and not is_market_hours(entry_time):
                current_idx += 1
                continue

            # Check if probability meets threshold
            if probability < self.probability_threshold:
                # No entry, move to next bar
                current_idx += 1
                continue

            # Check if trading is halted due to max drawdown
            if trading_halted:
                current_idx += 1
                continue

            # Entry signal found
            raw_entry_price = minute_bars.loc[current_idx, 'close']

            # Apply slippage to entry (worse entry = higher price for long)
            entry_price = raw_entry_price * (1 + self.slippage_percent / 100)

            # Simulate trade
            exit_idx = None
            exit_price = None
            exit_reason = None

            # Check each future bar until exit condition
            for future_idx in range(current_idx + 1, min(current_idx + self.prediction_horizon_minutes + 1, total_bars)):
                bar = minute_bars.loc[future_idx]
                high = bar['high']
                low = bar['low']
                close = bar['close']
                timestamp = bar['timestamp']

                # Calculate returns from actual entry price
                high_return = ((high - entry_price) / entry_price) * 100
                low_return = ((low - entry_price) / entry_price) * 100

                # Check stop loss FIRST (assume stop loss hit before target in same bar)
                if low_return <= -self.stop_loss_percent:
                    exit_price = entry_price * (1 - self.stop_loss_percent / 100)
                    exit_idx = future_idx
                    exit_reason = 'stop_loss'
                    break

                # Check if target hit
                if high_return >= self.target_percent:
                    exit_price = entry_price * (1 + self.target_percent / 100)
                    exit_idx = future_idx
                    exit_reason = 'target_hit'
                    break

                # Check if time limit reached
                elapsed_minutes = (timestamp - entry_time).total_seconds() / 60
                if elapsed_minutes >= self.prediction_horizon_minutes:
                    exit_price = close
                    exit_idx = future_idx
                    exit_reason = 'time_limit'
                    break

                # Check if market is about to close (force exit before 4:00 PM)
                if not is_market_hours(timestamp):
                    exit_price = close
                    exit_idx = future_idx
                    exit_reason = 'market_close'
                    break

            # If no exit found in loop, use last available bar
            if exit_idx is None:
                last_idx = min(current_idx + self.prediction_horizon_minutes, total_bars - 1)
                exit_idx = last_idx
                exit_price = minute_bars.loc[last_idx, 'close']
                exit_reason = 'time_limit'

            exit_time = minute_bars.loc[exit_idx, 'timestamp']

            # Apply slippage to exit (worse exit = lower price for sell)
            actual_exit_price = exit_price * (1 - self.slippage_percent / 100)

            # Calculate profit (net of commission, slippage already applied)
            gross_return = ((actual_exit_price - entry_price) / entry_price) * 100
            net_return = gross_return - self.commission_round_trip
            duration_minutes = (exit_time - entry_time).total_seconds() / 60

            # Create trade
            trade = Trade(
                ticker=ticker,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=actual_exit_price,
                exit_reason=exit_reason,
                probability=probability,
                profit_pct=net_return,
                duration_minutes=duration_minutes,
                model_type=model_type,
                target=target
            )
            result.add_trade(trade)

            # Update cumulative P&L and drawdown tracking
            cumulative_pnl += net_return
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            current_drawdown = peak_pnl - cumulative_pnl
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown

            # Check if max drawdown exceeded
            if current_drawdown >= max_drawdown_limit:
                trading_halted = True
                logger.warning(
                    f"Trading halted for {ticker}: Max drawdown {current_drawdown:.2f}% "
                    f"exceeded limit {max_drawdown_limit:.1f}%"
                )

            # IMMEDIATE RE-ENTRY: Move to the bar right after exit
            current_idx = exit_idx + 1

        # Add drawdown info to metadata
        result.metadata['cumulative_pnl'] = cumulative_pnl
        result.metadata['peak_pnl'] = peak_pnl
        result.metadata['max_drawdown'] = max_drawdown
        result.metadata['trading_halted'] = trading_halted

        logger.info(
            f"Re-entry backtest completed: {result.total_trades} trades "
            f"from {signals_checked} signals checked (immediate re-entry enabled, "
            f"max_drawdown={max_drawdown:.2f}%)"
        )

        return result

    def get_trade_outcomes(
        self,
        trades: List[Trade],
        probability_threshold: float = 0.5
    ) -> Tuple[List[bool], List[bool]]:
        """
        Convert trades to binary predictions and outcomes for accuracy calculation.

        Args:
            trades: List of trades
            probability_threshold: Threshold for positive prediction

        Returns:
            Tuple of (predictions, outcomes) as boolean lists
        """
        predictions = [t.probability >= probability_threshold for t in trades]
        outcomes = [t.is_win for t in trades]

        return predictions, outcomes

    def calculate_hit_rate(
        self,
        trades: List[Trade],
        hours: Optional[int] = None
    ) -> float:
        """
        Calculate hit rate (win rate) for trades.

        Args:
            trades: List of trades
            hours: Optional - only consider trades within last N hours

        Returns:
            Hit rate as float between 0 and 1
        """
        if not trades:
            return 0.0

        # Filter by time if specified
        if hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            trades = [t for t in trades if t.entry_time >= cutoff_time]

        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.is_win)
        return wins / len(trades)

    def export_trades(
        self,
        result: BacktestResult,
        filepath: str
    ) -> None:
        """
        Export backtest results to CSV.

        Args:
            result: BacktestResult to export
            filepath: Path to save CSV file
        """
        df = result.get_trades_df()
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} trades to {filepath}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestSimulator(threshold={self.probability_threshold:.1%}, "
            f"target={self.target_percent}%, stop_loss={self.stop_loss_percent}%, "
            f"horizon={self.prediction_horizon_minutes}min, "
            f"commission={self.commission_round_trip:.2%}, slippage={self.slippage_round_trip:.2%})"
        )
