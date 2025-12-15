"""Market hours utilities for NASDAQ trading."""

from datetime import datetime, time
from typing import Optional
import pytz

from config.settings import settings


# US Eastern timezone
ET = pytz.timezone('US/Eastern')


def get_market_hours() -> tuple[time, time]:
    """
    Get market open and close times.

    Returns:
        Tuple of (market_open_time, market_close_time) in ET
    """
    market_open = time(settings.MARKET_OPEN_HOUR, settings.MARKET_OPEN_MINUTE)
    market_close = time(settings.MARKET_CLOSE_HOUR, settings.MARKET_CLOSE_MINUTE)
    return market_open, market_close


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if the market is currently open.

    Args:
        dt: Datetime to check (default: current time)

    Returns:
        True if market is open, False otherwise
    """
    if dt is None:
        dt = datetime.now(ET)
    elif dt.tzinfo is None:
        dt = ET.localize(dt)
    else:
        dt = dt.astimezone(ET)

    # Check if it's a weekday (0=Monday, 6=Sunday)
    if dt.weekday() >= 5:
        return False

    market_open, market_close = get_market_hours()
    current_time = dt.time()

    return market_open <= current_time <= market_close


def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """
    Check if the given date is a trading day (weekday).

    Note: This does not account for US market holidays.

    Args:
        dt: Date to check (default: today)

    Returns:
        True if it's a weekday, False otherwise
    """
    if dt is None:
        dt = datetime.now(ET)

    return dt.weekday() < 5


def get_market_status() -> dict:
    """
    Get detailed market status.

    Returns:
        Dictionary with market status information
    """
    now_et = datetime.now(ET)
    market_open, market_close = get_market_hours()

    is_open = is_market_open(now_et)
    is_weekday = now_et.weekday() < 5

    status = {
        "is_open": is_open,
        "is_trading_day": is_weekday,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "market_open": market_open.strftime("%H:%M"),
        "market_close": market_close.strftime("%H:%M"),
    }

    if not is_open:
        if not is_weekday:
            status["reason"] = "Weekend"
        elif now_et.time() < market_open:
            status["reason"] = "Pre-market"
        else:
            status["reason"] = "After-hours"

    return status


def suppress_yfinance_warnings():
    """
    Suppress verbose yfinance warnings that occur outside market hours.

    Call this at application startup to reduce noise in logs.
    """
    import warnings
    import logging

    # Suppress yfinance peewee deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")

    # Reduce yfinance logging level
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    # Suppress urllib3 connection warnings
    logging.getLogger("urllib3").setLevel(logging.WARNING)
