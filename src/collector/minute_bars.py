"""Minute bar (OHLCV) data collection using Yahoo Finance (yfinance)."""

from typing import List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy import select, func

from config.settings import settings
from src.utils.database import (
    get_db,
    get_or_create_ticker,
    MinuteBar as DBMinuteBar,
)


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

    def __init__(self, use_db: bool = True):
        """
        Initialize minute bar collector.

        Args:
            use_db: If True, use database for incremental collection (default: True)
        """
        self.use_db = use_db
        logger.info(
            f"MinuteBarCollector initialized with Yahoo Finance "
            f"(incremental DB: {use_db})"
        )

    def get_latest_timestamp(self, ticker: str) -> Optional[datetime]:
        """
        Get the latest timestamp stored in database for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest timestamp as datetime or None if no data exists
        """
        if not self.use_db:
            return None

        try:
            with get_db() as db:
                stmt = (
                    select(func.max(DBMinuteBar.timestamp))
                    .where(DBMinuteBar.symbol == ticker)
                )
                result = db.execute(stmt).scalar_one_or_none()

                if result:
                    logger.debug(f"{ticker}: Latest stored timestamp: {result}")
                    return result
                else:
                    logger.debug(f"{ticker}: No existing data in database")
                    return None

        except Exception as e:
            logger.error(f"Error getting latest timestamp for {ticker}: {e}")
            return None

    def save_bars(self, bars: List[MinuteBar]) -> int:
        """
        Save minute bars to database.

        Args:
            bars: List of MinuteBar objects to save

        Returns:
            Number of bars successfully saved
        """
        if not self.use_db or not bars:
            return 0

        try:
            with get_db() as db:
                ticker_symbol = bars[0].ticker

                # Get or create ticker record
                ticker_record = get_or_create_ticker(db, ticker_symbol)

                saved_count = 0
                for bar in bars:
                    # Check if bar already exists (to avoid duplicates)
                    exists_stmt = (
                        select(DBMinuteBar)
                        .where(DBMinuteBar.symbol == bar.ticker)
                        .where(DBMinuteBar.timestamp == bar.datetime)
                    )
                    exists = db.execute(exists_stmt).scalar_one_or_none()

                    if not exists:
                        db_bar = DBMinuteBar(
                            ticker_id=ticker_record.id,
                            symbol=bar.ticker,
                            timestamp=bar.datetime,
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                            volume=int(bar.volume),
                            vwap=bar.vwap,
                            trade_count=bar.transactions,
                        )
                        db.add(db_bar)
                        saved_count += 1

                db.commit()

                if saved_count > 0:
                    logger.info(
                        f"Saved {saved_count} new bars for {ticker_symbol} to database"
                    )

                return saved_count

        except Exception as e:
            logger.error(f"Error saving bars to database: {e}")
            return 0

    def load_bars_from_db(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[MinuteBar]:
        """
        Load bars from database for a given time range.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date
            to_date: End date

        Returns:
            List of MinuteBar objects from database
        """
        if not self.use_db:
            return []

        try:
            with get_db() as db:
                stmt = (
                    select(DBMinuteBar)
                    .where(DBMinuteBar.symbol == ticker)
                    .where(DBMinuteBar.timestamp >= from_date)
                    .where(DBMinuteBar.timestamp <= to_date)
                    .order_by(DBMinuteBar.timestamp)
                )
                db_bars = db.execute(stmt).scalars().all()

                # Convert to dataclass MinuteBar
                bars = []
                for db_bar in db_bars:
                    bar = MinuteBar(
                        ticker=db_bar.symbol,
                        timestamp=int(db_bar.timestamp.timestamp() * 1000),
                        open=db_bar.open,
                        high=db_bar.high,
                        low=db_bar.low,
                        close=db_bar.close,
                        volume=float(db_bar.volume),
                        vwap=db_bar.vwap,
                        transactions=db_bar.trade_count,
                    )
                    bars.append(bar)

                if bars:
                    logger.debug(
                        f"Loaded {len(bars)} bars for {ticker} from database "
                        f"({from_date.date()} to {to_date.date()})"
                    )

                return bars

        except Exception as e:
            logger.error(f"Error loading bars from database: {e}")
            return []

    def get_bars(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        limit: int = 50000,
        force: bool = False
    ) -> List[MinuteBar]:
        """
        Get 1-minute bars for a ticker within a date range.

        Implements incremental data collection:
        1. Checks database for existing data
        2. Loads existing bars from DB
        3. Fetches only new data from Yahoo Finance (after latest DB timestamp)
        4. Saves new data to DB
        5. Returns combined dataset

        Args:
            ticker: Stock ticker symbol
            from_date: Start date
            to_date: End date
            limit: Maximum number of bars (not used in yfinance)
            force: If True, ignore existing data and fetch full range from API

        Returns:
            List of MinuteBar objects (from DB + newly fetched)

        Note:
            yfinance limits:
            - 1m interval: last 7 days only
            - 5m interval: last 60 days
        """
        try:
            # Force mode: skip incremental logic, fetch everything from API
            if force:
                logger.info(
                    f"{ticker}: Force mode - fetching full range from API "
                    f"({from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')})"
                )
                new_bars = self._fetch_from_yfinance(ticker, from_date, to_date)

                # Save to DB (duplicate checking is handled in save_bars)
                if new_bars and self.use_db:
                    self.save_bars(new_bars)

                logger.info(f"{ticker}: Force mode - fetched {len(new_bars)} bars")
                return new_bars

            # Step 1: Check for existing data in database
            latest_timestamp = self.get_latest_timestamp(ticker) if self.use_db else None

            # Step 2: Load existing data from database
            existing_bars = []
            if self.use_db and latest_timestamp:
                existing_bars = self.load_bars_from_db(ticker, from_date, to_date)

                # If we have all the data we need, return it
                if existing_bars and existing_bars[-1].datetime >= to_date:
                    logger.info(
                        f"{ticker}: All requested data ({len(existing_bars)} bars) "
                        f"available in database, no fetch needed"
                    )
                    return existing_bars

            # Step 3: Determine what new data to fetch
            fetch_from_date = from_date
            if latest_timestamp and latest_timestamp >= from_date:
                # Only fetch data after the latest timestamp
                fetch_from_date = latest_timestamp + timedelta(minutes=1)
                logger.info(
                    f"{ticker}: Found existing data up to {latest_timestamp}, "
                    f"fetching only new data from {fetch_from_date}"
                )

            # If fetch_from_date is already past to_date, we have all data
            if fetch_from_date >= to_date:
                logger.info(
                    f"{ticker}: Database already has all data up to {to_date}"
                )
                return existing_bars

            # Step 4: Fetch new data from Yahoo Finance
            new_bars = self._fetch_from_yfinance(ticker, fetch_from_date, to_date)

            # Step 5: Save new data to database
            if new_bars and self.use_db:
                self.save_bars(new_bars)

            # Step 6: Combine existing and new bars, remove duplicates
            all_bars = existing_bars + new_bars

            # Remove duplicates by timestamp (should not happen, but just in case)
            seen_timestamps = set()
            unique_bars = []
            for bar in all_bars:
                if bar.timestamp not in seen_timestamps:
                    seen_timestamps.add(bar.timestamp)
                    unique_bars.append(bar)

            # Sort by timestamp
            unique_bars.sort(key=lambda x: x.timestamp)

            logger.info(
                f"{ticker}: Total bars returned: {len(unique_bars)} "
                f"(existing: {len(existing_bars)}, new: {len(new_bars)})"
            )

            return unique_bars

        except Exception as e:
            logger.error(f"Error in get_bars for {ticker}: {e}")
            return []

    def _fetch_from_yfinance(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[MinuteBar]:
        """
        Internal method to fetch data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date
            to_date: End date

        Returns:
            List of MinuteBar objects
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
                f"Fetching {ticker} {interval} bars from Yahoo Finance: "
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
                logger.warning(f"No data returned from Yahoo Finance for {ticker}")
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
                f"Fetched {len(bars)} {interval} bars from Yahoo Finance for {ticker}"
            )

            return bars

        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance for {ticker}: {e}")
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
