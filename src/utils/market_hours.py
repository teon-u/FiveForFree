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


def get_last_market_close(dt: Optional[datetime] = None) -> datetime:
    """
    Get the datetime of the last market close.

    Args:
        dt: Reference datetime (default: current time)

    Returns:
        Datetime of the last market close in ET timezone
    """
    if dt is None:
        dt = datetime.now(ET)
    elif dt.tzinfo is None:
        dt = ET.localize(dt)
    else:
        dt = dt.astimezone(ET)

    _, market_close = get_market_hours()
    current_time = dt.time()

    # If today is a weekday and we're past market close, last close was today
    if dt.weekday() < 5 and current_time > market_close:
        last_close = dt.replace(
            hour=market_close.hour,
            minute=market_close.minute,
            second=0,
            microsecond=0
        )
    # Otherwise, find the most recent weekday's close
    else:
        # Start from yesterday
        check_date = dt.replace(
            hour=market_close.hour,
            minute=market_close.minute,
            second=0,
            microsecond=0
        )

        # If we're before market close today (or it's a weekday before close)
        if dt.weekday() < 5 and current_time <= market_close:
            # Go back one day
            from datetime import timedelta
            check_date = check_date - timedelta(days=1)

        # Go back to find a weekday
        from datetime import timedelta
        while check_date.weekday() >= 5:  # Saturday or Sunday
            check_date = check_date - timedelta(days=1)

        last_close = check_date

    return last_close


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
    last_close = get_last_market_close(now_et)

    status = {
        "is_open": is_open,
        "is_trading_day": is_weekday,
        "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        "market_open": market_open.strftime("%H:%M"),
        "market_close": market_close.strftime("%H:%M"),
        "last_close_et": last_close.strftime("%Y-%m-%d %H:%M:%S ET"),
        "last_close_iso": last_close.isoformat(),
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
