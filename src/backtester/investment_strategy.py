"""
Investment Strategy Module for FiveForFree Trading System

Implements enhanced trading strategies:
1. S+ Grade: win_rate >= 80% AND profit_factor >= 4.0
2. Rebalancing: Every 5 days recalculation
3. Transaction Costs: commission 0.25%, slippage 0.1%, stop loss -5%
4. Concentration Strategy: S+ -> top 2 stocks (50% each), else A-grade 4 stocks (25% each)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger


class StockGrade(Enum):
    """Stock grade classification based on performance metrics."""
    S_PLUS = "S+"   # win_rate >= 80% AND profit_factor >= 4.0
    A = "A"         # win_rate >= 70%
    B = "B"         # win_rate >= 60%
    C = "C"         # win_rate >= 50%
    D = "D"         # win_rate < 50%


@dataclass
class StockPerformance:
    """Performance metrics for a single stock."""
    ticker: str
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0
    grade: StockGrade = StockGrade.D

    def calculate_grade(self) -> StockGrade:
        """Calculate stock grade based on win rate and profit factor."""
        if self.win_rate >= 0.80 and self.profit_factor >= 4.0:
            self.grade = StockGrade.S_PLUS
        elif self.win_rate >= 0.70:
            self.grade = StockGrade.A
        elif self.win_rate >= 0.60:
            self.grade = StockGrade.B
        elif self.win_rate >= 0.50:
            self.grade = StockGrade.C
        else:
            self.grade = StockGrade.D
        return self.grade


@dataclass
class PortfolioAllocation:
    """Portfolio allocation for investment strategy."""
    allocations: Dict[str, float] = field(default_factory=dict)  # ticker -> weight (0-1)
    strategy: str = "balanced"  # "concentrated" or "balanced"
    rebalance_date: Optional[datetime] = None

    def get_position_size(self, ticker: str, total_capital: float) -> float:
        """Calculate position size for a ticker."""
        weight = self.allocations.get(ticker, 0.0)
        return total_capital * weight


@dataclass
class TransactionCosts:
    """Transaction cost parameters."""
    commission_pct: float = 0.25    # 0.25% one-way commission
    slippage_pct: float = 0.10      # 0.1% slippage per trade
    stop_loss_pct: float = 5.0      # 5% stop loss threshold

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip transaction cost."""
        return (self.commission_pct + self.slippage_pct) * 2


class InvestmentStrategy:
    """
    Enhanced investment strategy with S+ grade and concentration.

    Features:
    - S+ Grade: win_rate >= 80% AND profit_factor >= 4.0
    - Rebalancing: Every 5 days (configurable)
    - Transaction Costs: commission 0.25%, slippage 0.1%, stop loss -5%
    - Concentration: S+ stocks -> top 2 (50% each), else A-grade 4 stocks (25% each)
    """

    def __init__(
        self,
        rebalance_days: int = 5,
        transaction_costs: Optional[TransactionCosts] = None,
        min_trades_for_grade: int = 5
    ):
        """
        Initialize investment strategy.

        Args:
            rebalance_days: Days between portfolio rebalancing (default: 5)
            transaction_costs: Transaction cost parameters
            min_trades_for_grade: Minimum trades required to calculate grade
        """
        self.rebalance_days = rebalance_days
        self.costs = transaction_costs or TransactionCosts()
        self.min_trades_for_grade = min_trades_for_grade

        # Track last rebalance date
        self.last_rebalance: Optional[datetime] = None
        self.current_allocation: Optional[PortfolioAllocation] = None

        logger.info(
            f"InvestmentStrategy initialized: "
            f"rebalance={rebalance_days}d, "
            f"commission={self.costs.commission_pct}%, "
            f"slippage={self.costs.slippage_pct}%, "
            f"stop_loss={self.costs.stop_loss_pct}%"
        )

    def evaluate_stocks(
        self,
        performance_data: List[Dict]
    ) -> List[StockPerformance]:
        """
        Evaluate and grade stocks based on performance data.

        Args:
            performance_data: List of dicts with ticker, win_rate, profit_factor, etc.

        Returns:
            List of StockPerformance objects with grades
        """
        performances = []

        for data in performance_data:
            perf = StockPerformance(
                ticker=data.get('ticker', ''),
                win_rate=data.get('win_rate', 0.0),
                profit_factor=data.get('profit_factor', 0.0),
                total_return=data.get('total_return', 0.0),
                total_trades=data.get('total_trades', 0)
            )

            # Only grade stocks with enough trades
            if perf.total_trades >= self.min_trades_for_grade:
                perf.calculate_grade()

            performances.append(perf)

        # Sort by grade priority and then by profit factor
        grade_priority = {
            StockGrade.S_PLUS: 0,
            StockGrade.A: 1,
            StockGrade.B: 2,
            StockGrade.C: 3,
            StockGrade.D: 4
        }

        performances.sort(key=lambda x: (grade_priority[x.grade], -x.profit_factor))

        return performances

    def calculate_allocation(
        self,
        performances: List[StockPerformance]
    ) -> PortfolioAllocation:
        """
        Calculate portfolio allocation based on stock grades.

        Strategy:
        - If S+ stocks exist: Concentrate on top 2 S+ stocks (50% each)
        - If no S+: Distribute across top 4 A-grade stocks (25% each)

        Args:
            performances: List of StockPerformance objects (sorted by grade)

        Returns:
            PortfolioAllocation with weights
        """
        allocation = PortfolioAllocation(rebalance_date=datetime.now())

        # Get S+ and A-grade stocks
        s_plus_stocks = [p for p in performances if p.grade == StockGrade.S_PLUS]
        a_grade_stocks = [p for p in performances if p.grade == StockGrade.A]

        if s_plus_stocks:
            # Concentrated strategy: Top 2 S+ stocks
            top_stocks = s_plus_stocks[:2]
            weight = 1.0 / len(top_stocks)

            for stock in top_stocks:
                allocation.allocations[stock.ticker] = weight

            allocation.strategy = "concentrated"

            logger.info(
                f"Concentrated allocation on S+ stocks: "
                f"{[s.ticker for s in top_stocks]} ({weight*100:.1f}% each)"
            )

        elif a_grade_stocks:
            # Balanced strategy: Top 4 A-grade stocks
            top_stocks = a_grade_stocks[:4]
            weight = 1.0 / len(top_stocks)

            for stock in top_stocks:
                allocation.allocations[stock.ticker] = weight

            allocation.strategy = "balanced"

            logger.info(
                f"Balanced allocation on A-grade stocks: "
                f"{[s.ticker for s in top_stocks]} ({weight*100:.1f}% each)"
            )

        else:
            logger.warning("No S+ or A-grade stocks found for allocation")

        return allocation

    def should_rebalance(self, current_date: datetime) -> bool:
        """
        Check if portfolio should be rebalanced.

        Args:
            current_date: Current date to check

        Returns:
            True if rebalancing is needed
        """
        if self.last_rebalance is None:
            return True

        days_since_rebalance = (current_date - self.last_rebalance).days
        return days_since_rebalance >= self.rebalance_days

    def rebalance(
        self,
        current_date: datetime,
        performance_data: List[Dict]
    ) -> PortfolioAllocation:
        """
        Perform portfolio rebalancing.

        Args:
            current_date: Current date
            performance_data: Current performance data

        Returns:
            New portfolio allocation
        """
        performances = self.evaluate_stocks(performance_data)
        self.current_allocation = self.calculate_allocation(performances)
        self.last_rebalance = current_date

        logger.info(f"Portfolio rebalanced on {current_date.strftime('%Y-%m-%d')}")

        return self.current_allocation

    def calculate_trade_return(
        self,
        entry_price: float,
        exit_price: float,
        exit_reason: str = 'normal'
    ) -> float:
        """
        Calculate net return after transaction costs.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            exit_reason: 'normal', 'stop_loss', 'target_hit'

        Returns:
            Net return percentage
        """
        # Apply slippage
        actual_entry = entry_price * (1 + self.costs.slippage_pct / 100)
        actual_exit = exit_price * (1 - self.costs.slippage_pct / 100)

        # Calculate gross return
        gross_return = ((actual_exit - actual_entry) / actual_entry) * 100

        # Subtract commission
        net_return = gross_return - self.costs.round_trip_cost

        return net_return

    def apply_stop_loss(
        self,
        current_return: float
    ) -> Tuple[bool, float]:
        """
        Check and apply stop loss.

        Args:
            current_return: Current position return percentage

        Returns:
            Tuple of (is_stopped, adjusted_return)
        """
        if current_return <= -self.costs.stop_loss_pct:
            # Stop loss triggered
            return True, -self.costs.stop_loss_pct - self.costs.round_trip_cost

        return False, current_return

    def simulate_period(
        self,
        start_date: datetime,
        end_date: datetime,
        daily_returns: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Simulate investment over a period with rebalancing.

        Args:
            start_date: Start date
            end_date: End date
            daily_returns: Dict of ticker -> DataFrame with date and return columns
            initial_capital: Starting capital

        Returns:
            Dict with simulation results
        """
        current_date = start_date
        capital = initial_capital
        history = []

        while current_date <= end_date:
            # Check for rebalancing
            if self.should_rebalance(current_date):
                # Get performance data up to current date
                # (This would typically come from backtesting module)
                logger.info(f"Rebalancing check on {current_date.strftime('%Y-%m-%d')}")

            # Move to next day
            current_date += timedelta(days=1)

        return {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': ((capital - initial_capital) / initial_capital) * 100,
            'history': history
        }

    def get_grade_summary(
        self,
        performances: List[StockPerformance]
    ) -> Dict[str, List[str]]:
        """
        Get summary of stocks by grade.

        Args:
            performances: List of StockPerformance objects

        Returns:
            Dict with grade -> list of tickers
        """
        summary = {
            'S+': [],
            'A': [],
            'B': [],
            'C': [],
            'D': []
        }

        for perf in performances:
            summary[perf.grade.value].append(perf.ticker)

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InvestmentStrategy("
            f"rebalance_days={self.rebalance_days}, "
            f"costs={self.costs.round_trip_cost:.2f}%)"
        )


def calculate_enhanced_metrics(
    trades: List[Dict],
    apply_costs: bool = True,
    costs: Optional[TransactionCosts] = None
) -> Dict:
    """
    Calculate enhanced performance metrics with S+ grade evaluation.

    Args:
        trades: List of trade dictionaries
        apply_costs: Whether to apply transaction costs
        costs: Optional TransactionCosts object

    Returns:
        Dict with enhanced metrics including grade
    """
    if not trades:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_return': 0.0,
            'total_trades': 0,
            'grade': StockGrade.D.value
        }

    costs = costs or TransactionCosts()

    # Calculate basic metrics
    wins = [t for t in trades if t.get('profit_pct', 0) > 0]
    losses = [t for t in trades if t.get('profit_pct', 0) <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.0

    # Calculate profit factor
    gross_profit = sum(t.get('profit_pct', 0) for t in wins)
    gross_loss = abs(sum(t.get('profit_pct', 0) for t in losses))

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calculate total return
    total_return = sum(t.get('profit_pct', 0) for t in trades)

    # Apply transaction costs if needed
    if apply_costs:
        total_return -= len(trades) * costs.round_trip_cost

    # Determine grade
    if win_rate >= 0.80 and profit_factor >= 4.0:
        grade = StockGrade.S_PLUS.value
    elif win_rate >= 0.70:
        grade = StockGrade.A.value
    elif win_rate >= 0.60:
        grade = StockGrade.B.value
    elif win_rate >= 0.50:
        grade = StockGrade.C.value
    else:
        grade = StockGrade.D.value

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
        'total_return': total_return,
        'total_trades': len(trades),
        'grade': grade,
        'is_s_plus': grade == StockGrade.S_PLUS.value,
        'is_a_grade': grade in [StockGrade.S_PLUS.value, StockGrade.A.value]
    }
