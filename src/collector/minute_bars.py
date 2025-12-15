"""Minute bar (OHLCV) data collection module."""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import pandas as pd
from loguru import logger

from config.settings import settings
from src.collector.polygon_client import get_polygon_client, PolygonClientWrapper


@dataclass
class MinuteBar:
    """
    Represents a single minute-level OHLCV bar.

    Attributes:
        ticker: Stock ticker symbol
        timestamp: Bar timestamp (Unix milliseconds)
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        vwap: Volume-weighted average price
        transactions: Number of transactions
    """
    ticker: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    transactions: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime object."""
        return datetime.fromtimestamp(self.timestamp / 1000)


class MinuteBarCollector:
    """
    Collector for minute-level OHLCV bar data.

    Provides methods to fetch historical and real-time minute bars
    for individual tickers or batches.
    """

    def __init__(self, client: Optional[PolygonClientWrapper] = None):
        """
        Initialize minute bar collector.

        Args:
            client: Polygon client wrapper (uses global if not provided)
        """
        self.client = client or get_polygon_client()
        logger.info("MinuteBarCollector initialized")

    def get_bars(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        limit: int = 50000
    ) -> List[MinuteBar]:
        """
        Get minute bars for a ticker within a date range.

        Args:
            ticker: Stock ticker symbol
            from_date: Start datetime
            to_date: End datetime
            limit: Maximum number of bars to return

        Returns:
            List of MinuteBar objects
        """
        try:
            logger.debug(
                f"Fetching minute bars for {ticker}: "
                f"{from_date.strftime('%Y-%m-%d %H:%M')} to "
                f"{to_date.strftime('%Y-%m-%d %H:%M')}"
            )

            # Get raw bars from API
            raw_bars = self.client.get_minute_bars(ticker, from_date, to_date, limit)

            if not raw_bars:
                logger.debug(f"No bars found for {ticker}")
                return []

            # Convert to MinuteBar objects
            bars = []
            for bar in raw_bars:
                try:
                    minute_bar = MinuteBar(
                        ticker=ticker,
                        timestamp=int(bar.timestamp) if hasattr(bar, 'timestamp') else int(bar.t),
                        open=float(bar.open) if hasattr(bar, 'open') else float(bar.o),
                        high=float(bar.high) if hasattr(bar, 'high') else float(bar.h),
                        low=float(bar.low) if hasattr(bar, 'low') else float(bar.l),
                        close=float(bar.close) if hasattr(bar, 'close') else float(bar.c),
                        volume=float(bar.volume) if hasattr(bar, 'volume') else float(bar.v),
                        vwap=float(bar.vwap) if hasattr(bar, 'vwap') else float(getattr(bar, 'vw', 0)) or None,
                        transactions=int(bar.transactions) if hasattr(bar, 'transactions') else int(getattr(bar, 'n', 0)) or None
                    )
                    bars.append(minute_bar)
                except Exception as e:
                    logger.debug(f"Failed to parse bar: {str(e)}")
                    continue

            logger.info(f"Collected {len(bars)} minute bars for {ticker}")
            return bars

        except Exception as e:
            logger.error(f"Failed to get minute bars for {ticker}: {str(e)}")
            return []

    def get_recent_bars(
        self,
        ticker: str,
        minutes: int = 60
    ) -> List[MinuteBar]:
        """
        Get most recent N minutes of bars for a ticker.

        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes to look back

        Returns:
            List of MinuteBar objects
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(minutes=minutes)

        return self.get_bars(ticker, from_date, to_date)

    def get_today_bars(self, ticker: str) -> List[MinuteBar]:
        """
        Get all bars for today for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of MinuteBar objects
        """
        now = datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return self.get_bars(ticker, start_of_day, now)

    def get_market_hours_bars(
        self,
        ticker: str,
        date: Optional[datetime] = None
    ) -> List[MinuteBar]:
        """
        Get bars only during market hours (9:30 AM - 4:00 PM ET).

        Args:
            ticker: Stock ticker symbol
            date: Date to fetch (defaults to today)

        Returns:
            List of MinuteBar objects during market hours
        """
        if date is None:
            date = datetime.now()

        # Set market hours (9:30 AM - 4:00 PM ET)
        market_open = date.replace(
            hour=settings.MARKET_OPEN_HOUR,
            minute=settings.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = date.replace(
            hour=settings.MARKET_CLOSE_HOUR,
            minute=settings.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )

        # Get all bars for the day
        all_bars = self.get_bars(ticker, market_open, market_close)

        # Filter to market hours
        market_bars = [
            bar for bar in all_bars
            if market_open.timestamp() * 1000 <= bar.timestamp <= market_close.timestamp() * 1000
        ]

        logger.debug(
            f"Filtered {len(all_bars)} bars to {len(market_bars)} market hours bars for {ticker}"
        )

        return market_bars

    def get_bars_batch(
        self,
        tickers: List[str],
        from_date: datetime,
        to_date: datetime
    ) -> Dict[str, List[MinuteBar]]:
        """
        Get minute bars for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            from_date: Start datetime
            to_date: End datetime

        Returns:
            Dictionary mapping ticker to list of MinuteBar objects
        """
        results = {}

        logger.info(f"Fetching minute bars for {len(tickers)} tickers...")

        for ticker in tickers:
            bars = self.get_bars(ticker, from_date, to_date)
            if bars:
                results[ticker] = bars

        logger.info(f"Successfully fetched bars for {len(results)}/{len(tickers)} tickers")

        return results

    def to_dataframe(self, bars: List[MinuteBar]) -> pd.DataFrame:
        """
        Convert minute bars to pandas DataFrame.

        Args:
            bars: List of MinuteBar objects

        Returns:
            DataFrame with OHLCV data
        """
        if not bars:
            return pd.DataFrame()

        # Convert to list of dicts
        data = [bar.to_dict() for bar in bars]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set datetime as index
        df.set_index('datetime', inplace=True)

        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)

        return df

    def get_bars_as_dataframe(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime
    ) -> pd.DataFrame:
        """
        Get minute bars as DataFrame directly.

        Args:
            ticker: Stock ticker symbol
            from_date: Start datetime
            to_date: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        bars = self.get_bars(ticker, from_date, to_date)
        return self.to_dataframe(bars)

    def get_historical_bars(
        self,
        ticker: str,
        days: Optional[int] = None
    ) -> List[MinuteBar]:
        """
        Get historical minute bars for N days.

        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back (uses HISTORICAL_DAYS setting if not provided)

        Returns:
            List of MinuteBar objects
        """
        days = days or settings.HISTORICAL_DAYS
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        logger.info(f"Fetching {days} days of historical bars for {ticker}")

        return self.get_bars(ticker, from_date, to_date)

    def get_latest_bar(self, ticker: str) -> Optional[MinuteBar]:
        """
        Get the most recent completed minute bar for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest MinuteBar or None
        """
        # Get last 5 minutes to ensure we have data
        bars = self.get_recent_bars(ticker, minutes=5)

        if not bars:
            return None

        # Return most recent bar
        return max(bars, key=lambda x: x.timestamp)

    def calculate_bar_statistics(self, bars: List[MinuteBar]) -> Dict[str, float]:
        """
        Calculate statistics from a list of bars.

        Args:
            bars: List of MinuteBar objects

        Returns:
            Dictionary with statistical measures
        """
        if not bars:
            return {}

        df = self.to_dataframe(bars)

        stats = {
            'count': len(bars),
            'avg_volume': df['volume'].mean(),
            'total_volume': df['volume'].sum(),
            'avg_price': df['close'].mean(),
            'high': df['high'].max(),
            'low': df['low'].min(),
            'price_range': df['high'].max() - df['low'].min(),
            'avg_vwap': df['vwap'].mean() if 'vwap' in df.columns else None,
            'first_price': bars[0].open,
            'last_price': bars[-1].close,
            'price_change': bars[-1].close - bars[0].open,
            'price_change_pct': ((bars[-1].close - bars[0].open) / bars[0].open * 100)
                if bars[0].open > 0 else 0
        }

        return {k: v for k, v in stats.items() if v is not None}

    def validate_bars(self, bars: List[MinuteBar]) -> bool:
        """
        Validate bar data quality.

        Args:
            bars: List of MinuteBar objects

        Returns:
            True if valid, False otherwise
        """
        if not bars:
            logger.warning("Empty bar list")
            return False

        issues = []

        for i, bar in enumerate(bars):
            # Check for negative values
            if any([bar.open < 0, bar.high < 0, bar.low < 0, bar.close < 0, bar.volume < 0]):
                issues.append(f"Bar {i}: Negative values detected")

            # Check OHLC consistency
            if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
                issues.append(f"Bar {i}: OHLC inconsistency")

            # Check for zero volume (warning, not error)
            if bar.volume == 0:
                logger.debug(f"Bar {i}: Zero volume")

        if issues:
            logger.warning(f"Bar validation issues: {'; '.join(issues[:5])}")
            return False

        return True
