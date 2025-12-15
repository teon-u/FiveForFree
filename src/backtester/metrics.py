"""
Performance Metrics Module for NASDAQ Prediction System

Calculates comprehensive performance metrics for backtesting:
- Win rate, profit factor, Sharpe ratio
- Maximum drawdown, recovery time
- 50-hour rolling window metrics
- Per-model performance tracking

Integrates with model accuracy tracking and UI display.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from .simulator import Trade, BacktestResult

warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    Metrics include:
    - Basic: win rate, profit factor, total return
    - Risk-adjusted: Sharpe ratio, Sortino ratio, Calmar ratio
    - Drawdown: max drawdown, avg drawdown, recovery time
    - Trade stats: avg duration, best/worst trades
    """
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Return metrics
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown metrics
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0

    # Trade characteristics
    avg_duration_minutes: float = 0.0
    median_duration_minutes: float = 0.0
    target_hit_rate: float = 0.0
    time_exit_rate: float = 0.0

    # Time period
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    period_hours: float = 0.0

    # Additional stats
    expectancy: float = 0.0  # Expected value per trade
    kelly_criterion: float = 0.0  # Optimal position size
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Trade rate
    total_predictions: int = 0
    trade_rate: float = 0.0  # Percentage of predictions that became trades

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return_pct': self.total_return_pct,
            'avg_return_pct': self.avg_return_pct,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'best_trade_pct': self.best_trade_pct,
            'worst_trade_pct': self.worst_trade_pct,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'avg_drawdown_pct': self.avg_drawdown_pct,
            'max_drawdown_duration_hours': self.max_drawdown_duration_hours,
            'avg_duration_minutes': self.avg_duration_minutes,
            'median_duration_minutes': self.median_duration_minutes,
            'target_hit_rate': self.target_hit_rate,
            'time_exit_rate': self.time_exit_rate,
            'expectancy': self.expectancy,
            'kelly_criterion': self.kelly_criterion,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'period_hours': self.period_hours,
            'total_predictions': self.total_predictions,
            'trade_rate': self.trade_rate
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PerformanceMetrics(trades={self.total_trades}, "
            f"win_rate={self.win_rate:.1%}, return={self.total_return_pct:.2f}%, "
            f"sharpe={self.sharpe_ratio:.2f}, max_dd={self.max_drawdown_pct:.2f}%)"
        )


def calculate_metrics(
    trades: List[Trade],
    total_predictions: int = None,
    start_time: datetime = None,
    end_time: datetime = None
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from a list of trades.

    Args:
        trades: List of Trade objects
        total_predictions: Total number of predictions (for trade rate)
        start_time: Optional start time for period calculation
        end_time: Optional end time for period calculation

    Returns:
        PerformanceMetrics object with all calculated metrics
    """
    metrics = PerformanceMetrics()

    if not trades:
        return metrics

    # Basic counts
    metrics.total_trades = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.win_rate = metrics.winning_trades / metrics.total_trades

    # Return metrics
    returns = [t.profit_pct for t in trades]
    metrics.total_return_pct = sum(returns)
    metrics.avg_return_pct = np.mean(returns)
    metrics.avg_win_pct = np.mean([t.profit_pct for t in wins]) if wins else 0.0
    metrics.avg_loss_pct = np.mean([t.profit_pct for t in losses]) if losses else 0.0
    metrics.best_trade_pct = max(returns)
    metrics.worst_trade_pct = min(returns)

    # Profit factor
    gross_profit = sum(t.profit_pct for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.profit_pct for t in losses)) if losses else 0.0
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Risk-adjusted metrics
    if len(returns) > 1:
        returns_std = np.std(returns, ddof=1)
        if returns_std > 0:
            metrics.sharpe_ratio = metrics.avg_return_pct / returns_std
        else:
            metrics.sharpe_ratio = 0.0

        # Sortino ratio (uses only downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_std = np.std(downside_returns, ddof=1)
            if downside_std > 0:
                metrics.sortino_ratio = metrics.avg_return_pct / downside_std
            else:
                metrics.sortino_ratio = 0.0
        else:
            metrics.sortino_ratio = float('inf') if metrics.avg_return_pct > 0 else 0.0
    else:
        metrics.sharpe_ratio = 0.0
        metrics.sortino_ratio = 0.0

    # Drawdown metrics
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    metrics.max_drawdown_pct = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    metrics.avg_drawdown_pct = abs(np.mean(drawdowns[drawdowns < 0])) if any(drawdowns < 0) else 0.0

    # Calmar ratio (return / max drawdown)
    if metrics.max_drawdown_pct > 0:
        metrics.calmar_ratio = metrics.total_return_pct / metrics.max_drawdown_pct
    else:
        metrics.calmar_ratio = 0.0

    # Drawdown duration
    if len(drawdowns) > 1:
        in_drawdown = drawdowns < 0
        if any(in_drawdown):
            # Find longest consecutive drawdown period
            drawdown_starts = np.where(np.diff(np.concatenate([[False], in_drawdown])) == 1)[0]
            drawdown_ends = np.where(np.diff(np.concatenate([in_drawdown, [False]])) == -1)[0]

            if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                max_dd_duration = max(drawdown_ends - drawdown_starts)
                # Convert to hours (assuming trades are independent, use actual timing)
                if start_time and end_time:
                    total_hours = (end_time - start_time).total_seconds() / 3600
                    avg_trade_hours = total_hours / len(trades) if trades else 1
                    metrics.max_drawdown_duration_hours = max_dd_duration * avg_trade_hours

    # Trade duration metrics
    durations = [t.duration_minutes for t in trades]
    metrics.avg_duration_minutes = np.mean(durations)
    metrics.median_duration_minutes = np.median(durations)

    # Exit reason metrics
    target_hits = sum(1 for t in trades if t.exit_reason == 'target_hit')
    time_exits = sum(1 for t in trades if t.exit_reason == 'time_limit')
    metrics.target_hit_rate = target_hits / metrics.total_trades
    metrics.time_exit_rate = time_exits / metrics.total_trades

    # Expectancy (average profit per trade)
    metrics.expectancy = metrics.avg_return_pct

    # Kelly criterion for optimal position sizing
    if metrics.avg_loss_pct < 0:  # Avoid division by zero
        win_loss_ratio = abs(metrics.avg_win_pct / metrics.avg_loss_pct)
        metrics.kelly_criterion = (metrics.win_rate * win_loss_ratio - (1 - metrics.win_rate)) / win_loss_ratio
        metrics.kelly_criterion = max(0.0, min(metrics.kelly_criterion, 1.0))  # Clamp between 0 and 1
    else:
        metrics.kelly_criterion = 0.0

    # Consecutive wins/losses
    current_streak = 0
    current_type = None
    max_win_streak = 0
    max_loss_streak = 0

    for trade in trades:
        if trade.is_win:
            if current_type == 'win':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'win'
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_type == 'loss':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'loss'
            max_loss_streak = max(max_loss_streak, current_streak)

    metrics.max_consecutive_wins = max_win_streak
    metrics.max_consecutive_losses = max_loss_streak

    # Time period
    metrics.start_time = start_time or (min(t.entry_time for t in trades) if trades else None)
    metrics.end_time = end_time or (max(t.exit_time for t in trades) if trades else None)
    if metrics.start_time and metrics.end_time:
        metrics.period_hours = (metrics.end_time - metrics.start_time).total_seconds() / 3600

    # Trade rate
    if total_predictions:
        metrics.total_predictions = total_predictions
        metrics.trade_rate = metrics.total_trades / total_predictions

    return metrics


def calculate_rolling_metrics(
    trades: List[Trade],
    window_hours: int = 50
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics over a time window.

    Args:
        trades: List of Trade objects
        window_hours: Rolling window size in hours (default 50)

    Returns:
        DataFrame with rolling metrics indexed by time
    """
    if not trades:
        return pd.DataFrame()

    # Convert trades to DataFrame
    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

    # Create time-based rolling windows
    start_time = trades_df['entry_time'].min()
    end_time = trades_df['entry_time'].max()

    # Generate hourly timestamps
    time_points = pd.date_range(start=start_time, end=end_time, freq='1H')

    rolling_metrics = []

    for time_point in time_points:
        # Get trades within window
        window_start = time_point - timedelta(hours=window_hours)
        window_trades = trades_df[
            (trades_df['entry_time'] >= window_start) &
            (trades_df['entry_time'] <= time_point)
        ]

        if len(window_trades) == 0:
            continue

        # Calculate metrics for this window
        win_rate = window_trades['is_win'].mean()
        total_return = window_trades['profit_pct'].sum()
        avg_return = window_trades['profit_pct'].mean()
        num_trades = len(window_trades)

        # Sharpe ratio
        if num_trades > 1 and window_trades['profit_pct'].std() > 0:
            sharpe = avg_return / window_trades['profit_pct'].std()
        else:
            sharpe = 0.0

        rolling_metrics.append({
            'timestamp': time_point,
            'window_hours': window_hours,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return,
            'avg_return_pct': avg_return,
            'sharpe_ratio': sharpe
        })

    return pd.DataFrame(rolling_metrics)


class ModelPerformanceTracker:
    """
    Tracks performance metrics for each model type and target.

    Maintains 50-hour rolling metrics for UI display and model selection.
    Integrates with BacktestSimulator and model accuracy tracking.
    """

    def __init__(self, window_hours: int = None):
        """
        Initialize performance tracker.

        Args:
            window_hours: Rolling window size (default from settings)
        """
        self.window_hours = window_hours or settings.BACKTEST_HOURS

        # Store metrics by ticker -> model_type -> target -> metrics
        self.metrics_cache: Dict[str, Dict[str, Dict[str, PerformanceMetrics]]] = {}

        # Store recent trades for rolling calculations
        self.trades_cache: Dict[str, Dict[str, Dict[str, List[Trade]]]] = {}

        logger.info(f"ModelPerformanceTracker initialized with {self.window_hours}h window")

    def update_metrics(
        self,
        ticker: str,
        model_type: str,
        target: str,
        result: BacktestResult
    ) -> PerformanceMetrics:
        """
        Update metrics for a specific model.

        Args:
            ticker: Stock ticker
            model_type: Model type (xgboost, lightgbm, etc.)
            target: 'up' or 'down'
            result: BacktestResult with trades

        Returns:
            Updated PerformanceMetrics
        """
        # Initialize cache structure
        if ticker not in self.metrics_cache:
            self.metrics_cache[ticker] = {}
            self.trades_cache[ticker] = {}

        if model_type not in self.metrics_cache[ticker]:
            self.metrics_cache[ticker][model_type] = {}
            self.trades_cache[ticker][model_type] = {}

        # Store trades
        self.trades_cache[ticker][model_type][target] = result.trades

        # Calculate metrics
        metrics = calculate_metrics(
            trades=result.trades,
            total_predictions=result.total_predictions,
            start_time=result.start_time,
            end_time=result.end_time
        )

        # Cache metrics
        self.metrics_cache[ticker][model_type][target] = metrics

        logger.info(
            f"Updated metrics for {ticker} {model_type} {target}: "
            f"{metrics.total_trades} trades, {metrics.win_rate:.1%} win rate"
        )

        return metrics

    def get_metrics(
        self,
        ticker: str,
        model_type: str,
        target: str,
        hours: Optional[int] = None
    ) -> Optional[PerformanceMetrics]:
        """
        Get metrics for a specific model, optionally filtered by time window.

        Args:
            ticker: Stock ticker
            model_type: Model type
            target: 'up' or 'down'
            hours: Optional time window (default: use window_hours)

        Returns:
            PerformanceMetrics or None if not found
        """
        # Check if we have cached data
        if (ticker not in self.trades_cache or
            model_type not in self.trades_cache[ticker] or
            target not in self.trades_cache[ticker][model_type]):
            return None

        trades = self.trades_cache[ticker][model_type][target]

        if not trades:
            return None

        # Filter by time window if specified
        if hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            trades = [t for t in trades if t.entry_time >= cutoff_time]

        if not trades:
            return None

        # Calculate and return metrics
        return calculate_metrics(trades)

    def get_recent_hit_rate(
        self,
        ticker: str,
        model_type: str,
        target: str,
        hours: Optional[int] = None
    ) -> float:
        """
        Get recent hit rate (win rate) for a model.

        This is the key metric used by ModelManager to select best model.

        Args:
            ticker: Stock ticker
            model_type: Model type
            target: 'up' or 'down'
            hours: Time window (default: use window_hours)

        Returns:
            Hit rate as float between 0 and 1
        """
        metrics = self.get_metrics(ticker, model_type, target, hours or self.window_hours)
        return metrics.win_rate if metrics else 0.0

    def get_all_model_performances(
        self,
        ticker: str,
        hours: Optional[int] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get performance summary for all models of a ticker.

        Used by UI to display model comparison table.

        Args:
            ticker: Stock ticker
            hours: Time window (default: use window_hours)

        Returns:
            Dictionary: {target: {model_type: {metric: value}}}
        """
        hours = hours or self.window_hours
        performances = {'up': {}, 'down': {}}

        if ticker not in self.trades_cache:
            return performances

        for target in ['up', 'down']:
            for model_type in settings.MODEL_TYPES:
                metrics = self.get_metrics(ticker, model_type, target, hours)

                if metrics:
                    performances[target][model_type] = {
                        'hit_rate': metrics.win_rate * 100,  # Convert to percentage
                        'total_trades': metrics.total_trades,
                        'total_return': metrics.total_return_pct,
                        'profit_factor': metrics.profit_factor,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'is_trained': metrics.total_trades > 0
                    }
                else:
                    performances[target][model_type] = {
                        'hit_rate': 0.0,
                        'total_trades': 0,
                        'total_return': 0.0,
                        'profit_factor': 0.0,
                        'sharpe_ratio': 0.0,
                        'is_trained': False
                    }

        return performances

    def get_best_model(
        self,
        ticker: str,
        target: str = 'up',
        hours: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """
        Get the best performing model for a ticker and target.

        Args:
            ticker: Stock ticker
            target: 'up' or 'down'
            hours: Time window (default: use window_hours)

        Returns:
            Tuple of (best_model_type, hit_rate)
        """
        hours = hours or self.window_hours

        best_model = None
        best_hit_rate = -1.0

        if ticker not in self.trades_cache:
            return None, 0.0

        for model_type in settings.MODEL_TYPES:
            hit_rate = self.get_recent_hit_rate(ticker, model_type, target, hours)

            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_model = model_type

        return best_model, best_hit_rate

    def get_rolling_performance(
        self,
        ticker: str,
        model_type: str,
        target: str
    ) -> pd.DataFrame:
        """
        Get rolling performance metrics over time.

        Args:
            ticker: Stock ticker
            model_type: Model type
            target: 'up' or 'down'

        Returns:
            DataFrame with rolling metrics
        """
        if (ticker not in self.trades_cache or
            model_type not in self.trades_cache[ticker] or
            target not in self.trades_cache[ticker][model_type]):
            return pd.DataFrame()

        trades = self.trades_cache[ticker][model_type][target]
        return calculate_rolling_metrics(trades, self.window_hours)

    def export_summary(
        self,
        ticker: str,
        filepath: str
    ) -> None:
        """
        Export performance summary to CSV.

        Args:
            ticker: Stock ticker
            filepath: Path to save CSV
        """
        if ticker not in self.trades_cache:
            logger.warning(f"No data for ticker {ticker}")
            return

        summary_rows = []

        for model_type in settings.MODEL_TYPES:
            for target in ['up', 'down']:
                metrics = self.get_metrics(ticker, model_type, target, self.window_hours)

                if metrics:
                    row = {
                        'ticker': ticker,
                        'model_type': model_type,
                        'target': target,
                        **metrics.to_dict()
                    }
                    summary_rows.append(row)

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported performance summary to {filepath}")
        else:
            logger.warning(f"No metrics to export for {ticker}")

    def clear_old_trades(self, days: int = 7) -> None:
        """
        Remove trades older than specified days to save memory.

        Args:
            days: Number of days to keep
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        for ticker in self.trades_cache:
            for model_type in self.trades_cache[ticker]:
                for target in self.trades_cache[ticker][model_type]:
                    old_count = len(self.trades_cache[ticker][model_type][target])
                    self.trades_cache[ticker][model_type][target] = [
                        t for t in self.trades_cache[ticker][model_type][target]
                        if t.entry_time >= cutoff_time
                    ]
                    new_count = len(self.trades_cache[ticker][model_type][target])

                    if old_count != new_count:
                        logger.info(
                            f"Cleared {old_count - new_count} old trades for "
                            f"{ticker} {model_type} {target}"
                        )

    def __repr__(self) -> str:
        """String representation."""
        total_tickers = len(self.trades_cache)
        total_models = sum(
            len(models)
            for models in self.trades_cache.values()
        )
        return f"ModelPerformanceTracker(window={self.window_hours}h, tickers={total_tickers}, models={total_models})"
