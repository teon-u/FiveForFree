"""
Risk Management System for Position Sizing and Portfolio Risk Control

This module implements various position sizing strategies and risk management rules:
- Kelly Criterion: Mathematical optimal betting ratio
- Fixed Fractional: Risk a fixed percentage per trade
- ATR Position Sizer: Volatility-based position sizing
- Daily Risk Manager: Daily loss limits and cooldown
- Portfolio Limits: Max positions and exposure limits
"""

from typing import Dict, Optional, Tuple
from loguru import logger


class KellyCriterion:
    """
    Kelly Criterion position sizing strategy.

    Calculates optimal position size based on win rate and reward/risk ratio.
    Uses Half-Kelly by default to reduce volatility.
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Initialize Kelly Criterion calculator.

        Args:
            kelly_fraction: Kelly fraction to use (0.5 = Half-Kelly, recommended)
        """
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        account_balance: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Formula: f* = (p × b - q) / b
        where:
            f* = optimal fraction of capital to bet
            p = win rate
            q = loss rate (1 - p)
            b = reward/risk ratio

        Args:
            account_balance: Current account balance
            win_rate: Win rate (0.0 to 1.0)
            avg_win_pct: Average winning trade percentage
            avg_loss_pct: Average losing trade percentage (positive number)

        Returns:
            Position size in dollars
        """
        if avg_loss_pct == 0:
            logger.warning("avg_loss_pct is zero, cannot calculate Kelly")
            return 0

        # Calculate reward/risk ratio
        reward_risk_ratio = avg_win_pct / avg_loss_pct

        # Kelly formula
        kelly = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio

        # Don't bet if Kelly is negative
        if kelly <= 0:
            logger.debug(f"Kelly is negative ({kelly:.4f}), position size = 0")
            return 0

        # Apply Kelly fraction (Half-Kelly)
        kelly_adjusted = kelly * self.kelly_fraction

        # Cap at 25% to avoid over-leveraging
        kelly_capped = min(kelly_adjusted, 0.25)

        position_size = account_balance * kelly_capped

        logger.debug(
            f"Kelly: raw={kelly:.4f}, adjusted={kelly_adjusted:.4f}, "
            f"capped={kelly_capped:.4f}, size=${position_size:.2f}"
        )

        return position_size


class FixedFractional:
    """
    Fixed Fractional position sizing strategy.

    Risks a fixed percentage of account balance per trade.
    Simple and effective for most traders.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.25
    ):
        """
        Initialize Fixed Fractional calculator.

        Args:
            risk_per_trade: Percentage of account to risk per trade (e.g., 0.01 = 1%)
            max_position_pct: Maximum position size as percentage of account (e.g., 0.25 = 25%)
        """
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_pct: float
    ) -> float:
        """
        Calculate position size using Fixed Fractional method.

        Formula: Position Size = (Account × Risk%) / Stop Loss%

        Args:
            account_balance: Current account balance
            stop_loss_pct: Stop loss percentage (positive number, e.g., 0.5 for 0.5%)

        Returns:
            Position size in dollars
        """
        if stop_loss_pct <= 0:
            logger.warning("stop_loss_pct must be positive")
            return 0

        # Maximum loss amount we're willing to accept
        max_loss_amount = account_balance * self.risk_per_trade

        # Calculate position size
        position_size = max_loss_amount / (stop_loss_pct / 100)

        # Apply maximum position limit
        max_position = account_balance * self.max_position_pct
        position_size = min(position_size, max_position)

        logger.debug(
            f"FixedFractional: max_loss=${max_loss_amount:.2f}, "
            f"stop_loss={stop_loss_pct}%, size=${position_size:.2f}"
        )

        return position_size


class ATRPositionSizer:
    """
    ATR-based position sizing strategy.

    Adjusts position size based on market volatility (ATR).
    Higher volatility = smaller position size.
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        atr_multiplier: float = 2.0,
        atr_period: int = 14
    ):
        """
        Initialize ATR Position Sizer.

        Args:
            risk_per_trade: Percentage of account to risk per trade
            atr_multiplier: ATR multiplier for stop loss distance
            atr_period: ATR calculation period
        """
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        atr: float
    ) -> float:
        """
        Calculate position size based on ATR.

        Args:
            account_balance: Current account balance
            current_price: Current stock price
            atr: Average True Range value

        Returns:
            Position size in dollars
        """
        if atr <= 0 or current_price <= 0:
            logger.warning("Invalid ATR or price")
            return 0

        # ATR-based stop distance (in price terms)
        stop_distance = atr * self.atr_multiplier

        # Maximum loss amount
        max_loss_amount = account_balance * self.risk_per_trade

        # Calculate number of shares
        shares = max_loss_amount / stop_distance

        # Position size in dollars
        position_size = shares * current_price

        # Cap at 25% of account
        max_position = account_balance * 0.25
        position_size = min(position_size, max_position)

        logger.debug(
            f"ATR: atr={atr:.2f}, stop_dist={stop_distance:.2f}, "
            f"shares={shares:.0f}, size=${position_size:.2f}"
        )

        return position_size

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """
        Calculate stop loss price based on ATR.

        Args:
            entry_price: Entry price
            atr: Average True Range

        Returns:
            Stop loss price
        """
        return entry_price - (atr * self.atr_multiplier)


class PositionLimits:
    """Position size and exposure limits."""

    # Single position maximum (% of account)
    MAX_SINGLE_POSITION = 0.10  # 10%

    # Sector exposure maximum (% of account)
    MAX_SECTOR_EXPOSURE = 0.30  # 30%

    # Total portfolio exposure maximum (% of account)
    MAX_TOTAL_EXPOSURE = 0.80  # 80% (keep 20% cash)


class PortfolioLimits:
    """Portfolio-level limits and constraints."""

    def __init__(
        self,
        max_positions: int = 5,
        min_position_size: float = 1000
    ):
        """
        Initialize portfolio limits.

        Args:
            max_positions: Maximum number of concurrent positions
            min_position_size: Minimum position size in dollars
        """
        self.max_positions = max_positions
        self.min_position_size = min_position_size

    def can_open_position(
        self,
        current_positions: int,
        position_size: float
    ) -> Tuple[bool, str]:
        """
        Check if new position can be opened.

        Args:
            current_positions: Current number of open positions
            position_size: Proposed position size

        Returns:
            (can_open, reason)
        """
        if current_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        if position_size < self.min_position_size:
            return False, f"Position too small (${position_size:.0f} < ${self.min_position_size})"

        return True, "OK"


class DailyRiskManager:
    """
    Daily risk management with loss limits and cooldown.

    Prevents excessive losses on bad days and enforces cooling-off
    periods after consecutive losses.
    """

    def __init__(
        self,
        max_daily_loss: float = 0.03,
        max_trades_per_day: int = 10,
        cooldown_after_loss: int = 2
    ):
        """
        Initialize Daily Risk Manager.

        Args:
            max_daily_loss: Maximum daily loss as % of account (e.g., 0.03 = 3%)
            max_trades_per_day: Maximum number of trades per day
            cooldown_after_loss: Number of trades to skip after 3 consecutive losses
        """
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_after_loss = cooldown_after_loss

        # Daily state
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_remaining = 0

    def record_trade(self, profit_pct: float):
        """
        Record a trade result.

        Args:
            profit_pct: Trade profit percentage
        """
        self.daily_pnl += profit_pct
        self.trades_today += 1

        if profit_pct < 0:
            self.consecutive_losses += 1

            # Trigger cooldown after 3 consecutive losses
            if self.consecutive_losses >= 3:
                self.cooldown_remaining = self.cooldown_after_loss
                logger.warning(
                    f"3 consecutive losses detected. "
                    f"Cooldown activated for {self.cooldown_after_loss} trades."
                )
        else:
            self.consecutive_losses = 0

    def can_trade(self, account_balance: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Args:
            account_balance: Current account balance

        Returns:
            (can_trade, reason)
        """
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss * 100:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f}%)"

        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day reached ({self.max_trades_per_day})"

        # Check cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False, f"Cooldown active ({self.cooldown_remaining} trades remaining)"

        return True, "OK"

    def reset_daily(self):
        """Reset daily counters (call at start of each day)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.cooldown_remaining = 0
        logger.info("Daily risk manager reset")


class RiskManager:
    """
    Integrated risk management system.

    Combines position sizing, portfolio limits, and daily risk management
    into a unified interface.
    """

    def __init__(
        self,
        account_balance: float,
        risk_method: str = "fixed_fractional",
        risk_per_trade: float = 0.01,
        max_daily_loss: float = 0.03,
        max_positions: int = 5,
        min_position_size: float = 1000
    ):
        """
        Initialize Risk Manager.

        Args:
            account_balance: Initial account balance
            risk_method: Position sizing method ('kelly', 'fixed_fractional', 'atr')
            risk_per_trade: Risk per trade as % of account
            max_daily_loss: Max daily loss as % of account
            max_positions: Maximum concurrent positions
            min_position_size: Minimum position size in dollars
        """
        self.account_balance = account_balance
        self.risk_method = risk_method

        # Position sizing strategies
        self.kelly = KellyCriterion(kelly_fraction=0.5)
        self.fixed = FixedFractional(risk_per_trade=risk_per_trade)
        self.atr_sizer = ATRPositionSizer(risk_per_trade=risk_per_trade)

        # Risk limits
        self.position_limits = PositionLimits()
        self.portfolio_limits = PortfolioLimits(
            max_positions=max_positions,
            min_position_size=min_position_size
        )
        self.daily_risk = DailyRiskManager(max_daily_loss=max_daily_loss)

        # Portfolio state
        self.current_positions: Dict[str, dict] = {}

    def calculate_position_size(
        self,
        ticker: str,
        current_price: float,
        stop_loss_pct: float = 0.5,
        win_rate: Optional[float] = None,
        avg_win_pct: Optional[float] = None,
        avg_loss_pct: Optional[float] = None,
        atr: Optional[float] = None
    ) -> dict:
        """
        Calculate position size with all risk checks applied.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            stop_loss_pct: Stop loss percentage
            win_rate: Win rate (for Kelly)
            avg_win_pct: Average win percentage (for Kelly)
            avg_loss_pct: Average loss percentage (for Kelly)
            atr: Average True Range (for ATR sizing)

        Returns:
            Dictionary with:
                - can_trade: bool
                - position_size: float
                - shares: int
                - reason: str
        """
        result = {
            'can_trade': False,
            'position_size': 0,
            'shares': 0,
            'reason': ''
        }

        # 1. Check daily risk limits
        can_trade, reason = self.daily_risk.can_trade(self.account_balance)
        if not can_trade:
            result['reason'] = reason
            return result

        # 2. Check portfolio limits
        can_open, reason = self.portfolio_limits.can_open_position(
            len(self.current_positions),
            0  # We'll check actual size later
        )
        if not can_open and "Max positions" in reason:
            result['reason'] = reason
            return result

        # 3. Calculate raw position size based on selected method
        if self.risk_method == "kelly" and all([win_rate, avg_win_pct, avg_loss_pct]):
            raw_size = self.kelly.calculate_position_size(
                self.account_balance,
                win_rate,
                avg_win_pct,
                avg_loss_pct
            )
        elif self.risk_method == "atr" and atr:
            raw_size = self.atr_sizer.calculate_position_size(
                self.account_balance,
                current_price,
                atr
            )
        else:
            # Default to Fixed Fractional
            raw_size = self.fixed.calculate_position_size(
                self.account_balance,
                stop_loss_pct
            )

        # 4. Apply maximum single position limit
        max_single = self.account_balance * self.position_limits.MAX_SINGLE_POSITION
        position_size = min(raw_size, max_single)

        # 5. Check minimum position size
        if position_size < self.portfolio_limits.min_position_size:
            result['reason'] = (
                f"Position too small (${position_size:.0f} < "
                f"${self.portfolio_limits.min_position_size})"
            )
            return result

        # 6. Calculate number of shares
        shares = int(position_size / current_price)
        if shares <= 0:
            result['reason'] = "Cannot afford even 1 share"
            return result

        # 7. Recalculate exact position size based on whole shares
        final_position_size = shares * current_price

        # Success
        result['can_trade'] = True
        result['position_size'] = final_position_size
        result['shares'] = shares
        result['reason'] = "OK"

        logger.info(
            f"Position sizing for {ticker}: "
            f"${final_position_size:.2f} ({shares} shares @ ${current_price:.2f})"
        )

        return result

    def open_position(self, ticker: str, shares: int, entry_price: float):
        """
        Record a position opening.

        Args:
            ticker: Stock ticker
            shares: Number of shares
            entry_price: Entry price
        """
        self.current_positions[ticker] = {
            'shares': shares,
            'entry_price': entry_price,
            'position_value': shares * entry_price
        }
        logger.info(
            f"Position opened: {ticker} - {shares} shares @ ${entry_price:.2f}"
        )

    def close_position(self, ticker: str, exit_price: float, profit_pct: float):
        """
        Record a position closing.

        Args:
            ticker: Stock ticker
            exit_price: Exit price
            profit_pct: Profit percentage
        """
        if ticker in self.current_positions:
            pos = self.current_positions[ticker]
            logger.info(
                f"Position closed: {ticker} - {pos['shares']} shares @ ${exit_price:.2f} "
                f"({profit_pct:+.2f}%)"
            )
            del self.current_positions[ticker]

        # Record trade for daily risk management
        self.daily_risk.record_trade(profit_pct)

    def update_balance(self, new_balance: float):
        """
        Update account balance.

        Args:
            new_balance: New account balance
        """
        old_balance = self.account_balance
        self.account_balance = new_balance
        logger.info(
            f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f} "
            f"({((new_balance/old_balance - 1) * 100):+.2f}%)"
        )

    def get_status(self) -> dict:
        """
        Get current risk manager status.

        Returns:
            Status dictionary
        """
        return {
            'account_balance': self.account_balance,
            'current_positions': len(self.current_positions),
            'max_positions': self.portfolio_limits.max_positions,
            'daily_pnl': self.daily_risk.daily_pnl,
            'trades_today': self.daily_risk.trades_today,
            'consecutive_losses': self.daily_risk.consecutive_losses,
            'cooldown_remaining': self.daily_risk.cooldown_remaining
        }
