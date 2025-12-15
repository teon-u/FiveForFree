"""Minute bar (OHLCV) data collection using Yahoo Finance (yfinance)."""

from typing import List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import pandas as pd
import yfinance as yf
from loguru import logger

from config.settings import settings


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
        vwap: Volume-weighted average price (calculated)
        transactions: Number of transactions (not available)
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
    Collector for minute-level OHLCV bar data using Yahoo Finance.

    Yahoo Finance provides:
    - 1-minute bars (last 7 days)
    - 5-minute bars (last 60 days)
    - Completely free, no API key required

    Note: Uses 1-minute intervals for maximum resolution.
    """

    def __init__(self):
        """Initialize minute bar collector."""
        logger.info("MinuteBarCollector initialized with Yahoo Finance")

    def get_bars(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        limit: int = 50000
    ) -> List[MinuteBar]:
        """
        Get 1-minute bars for a ticker within a date range.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date
            to_date: End date
            limit: Maximum number of bars (not used in yfinance)

        Returns:
            List of MinuteBar objects

        Note:
            yfinance limits:
            - 1m interval: last 7 days only
            - 5m interval: last 60 days
        """
        try:
            # Calculate days difference
            days_diff = (to_date - from_date).days

            # Choose interval based on range
            if days_diff <= 7:
                interval = '1m'
            elif days_diff <= 60:
                interval = '5m'
            else:
                logger.warning(
                    f"{ticker}: Range {days_diff} days exceeds yfinance limit. "
                    "Using 5m interval for last 60 days."
                )
                from_date = to_date - timedelta(days=60)
                interval = '5m'

            logger.debug(
                f"Fetching {ticker} {interval} bars: "
                f"{from_date.date()} to {to_date.date()}"
            )

            # Download data from Yahoo Finance
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(
                start=from_date,
                end=to_date,
                interval=interval,
                prepost=False  # Regular market hours only
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return []

            # Convert to MinuteBar objects
            bars = []
            for idx, row in df.iterrows():
                # Calculate VWAP as (H+L+C)/3
                vwap = (row['High'] + row['Low'] + row['Close']) / 3

                bar = MinuteBar(
                    ticker=ticker,
                    timestamp=int(idx.timestamp() * 1000),  # Convert to milliseconds
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    vwap=float(vwap),
                    transactions=None  # Not available
                )
                bars.append(bar)

            logger.info(
                f"Retrieved {len(bars)} {interval} bars for {ticker} "
                f"({from_date.date()} to {to_date.date()})"
            )

            return bars

        except Exception as e:
            logger.error(f"Error fetching bars for {ticker}: {e}")
            return []

    def get_recent_bars(self, ticker: str, minutes: int = 60) -> List[MinuteBar]:
        """
        Get the most recent N minutes of bar data.

        Args:
            ticker: Stock ticker symbol
            minutes: Number of minutes to fetch (max 7 days worth)

        Returns:
            List of MinuteBar objects
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(minutes=minutes + 30)  # Buffer for market hours

        bars = self.get_bars(ticker, from_date, to_date)

        # Return only the last N minutes
        return bars[-minutes:] if bars else []

    def get_bars_as_dataframe(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime
    ) -> pd.DataFrame:
        """
        Get bars as a pandas DataFrame.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date
            to_date: End date

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        bars = self.get_bars(ticker, from_date, to_date)

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame([bar.to_dict() for bar in bars])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df

    def get_bars_batch(
        self,
        tickers: List[str],
        from_date: datetime,
        to_date: datetime
    ) -> dict[str, List[MinuteBar]]:
        """
        Get bars for multiple tickers.

        Args:
            tickers: List of ticker symbols
            from_date: Start date
            to_date: End date

        Returns:
            Dictionary mapping ticker to list of MinuteBar objects
        """
        result = {}

        for ticker in tickers:
            bars = self.get_bars(ticker, from_date, to_date)
            if bars:
                result[ticker] = bars

        logger.info(
            f"Batch collection complete: {len(result)}/{len(tickers)} tickers"
        )

        return result

    def get_latest_bar(self, ticker: str) -> Optional[MinuteBar]:
        """
        Get the most recent bar for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest MinuteBar or None if unavailable
        """
        bars = self.get_recent_bars(ticker, minutes=10)
        return bars[-1] if bars else None

    def calculate_bar_statistics(self, bars: List[MinuteBar]) -> dict:
        """
        Calculate statistics from a list of bars.

        Args:
            bars: List of MinuteBar objects

        Returns:
            Dictionary with statistics
        """
        if not bars:
            return {}

        prices = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]

        return {
            'count': len(bars),
            'avg_price': sum(prices) / len(prices),
            'high': max(bar.high for bar in bars),
            'low': min(bar.low for bar in bars),
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes),
            'price_change_pct': ((prices[-1] - prices[0]) / prices[0] * 100)
            if prices[0] > 0 else 0
        }
