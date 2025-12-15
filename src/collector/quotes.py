"""Quote data collection module using Finnhub API."""

from typing import Optional, Dict
from datetime import datetime

from loguru import logger

from src.collector.finnhub_client import get_finnhub_client, FinnhubClientWrapper


class QuoteCollector:
    """
    Collector for basic quote data using Finnhub API.

    Provides methods to fetch real-time quote information.
    Note: Level 2 order book data is not available on Finnhub free tier.
    """

    def __init__(self, client: Optional[FinnhubClientWrapper] = None):
        """
        Initialize quote collector.

        Args:
            client: Finnhub client wrapper (uses global if not provided)
        """
        self.client = client or get_finnhub_client()
        logger.info("QuoteCollector initialized with Finnhub")

    def get_quote(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get basic quote for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with quote data:
            - current_price: Current price
            - high: Day's high price
            - low: Day's low price
            - open: Opening price
            - prev_close: Previous close price
            - timestamp: Quote timestamp (Unix seconds)
        """
        try:
            logger.debug(f"Fetching quote for {ticker}")

            # Get quote from Finnhub
            quote_data = self.client.get_quote(ticker)

            if not quote_data:
                logger.debug(f"No quote data available for {ticker}")
                return None

            # Extract quote information
            # Finnhub quote keys: c (current), h (high), l (low), o (open), pc (prev close), t (timestamp)
            current_price = quote_data.get('c')
            high = quote_data.get('h')
            low = quote_data.get('l')
            open_price = quote_data.get('o')
            prev_close = quote_data.get('pc')
            timestamp = quote_data.get('t')

            # Validate required fields
            if current_price is None or current_price <= 0:
                logger.debug(f"Invalid quote data for {ticker}")
                return None

            quote = {
                'current_price': float(current_price),
                'high': float(high) if high is not None else float(current_price),
                'low': float(low) if low is not None else float(current_price),
                'open': float(open_price) if open_price is not None else float(current_price),
                'prev_close': float(prev_close) if prev_close is not None else float(current_price),
                'timestamp': int(timestamp) if timestamp is not None else int(datetime.now().timestamp())
            }

            logger.debug(
                f"{ticker}: ${quote['current_price']:.2f} "
                f"(H: ${quote['high']:.2f}, L: ${quote['low']:.2f})"
            )

            return quote

        except Exception as e:
            logger.error(f"Failed to get quote for {ticker}: {str(e)}")
            return None

    def get_quotes_batch(self, tickers: list[str]) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Get quotes for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker to quote data
        """
        results = {}

        logger.info(f"Fetching quotes for {len(tickers)} tickers...")

        for ticker in tickers:
            quote = self.get_quote(ticker)
            results[ticker] = quote

        successful = sum(1 for q in results.values() if q is not None)
        logger.info(f"Successfully fetched quotes for {successful}/{len(tickers)} tickers")

        return results

    def get_price_change(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get price change information for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with:
            - current_price: Current price
            - prev_close: Previous close
            - change: Absolute change
            - change_percent: Percentage change
        """
        quote = self.get_quote(ticker)

        if not quote:
            return None

        current_price = quote['current_price']
        prev_close = quote['prev_close']

        if prev_close <= 0:
            return None

        change = current_price - prev_close
        change_percent = (change / prev_close) * 100

        return {
            'current_price': current_price,
            'prev_close': prev_close,
            'change': change,
            'change_percent': change_percent
        }

    def get_spread_estimate(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Estimate bid-ask spread using high/low range.

        Note: This is an approximation as Finnhub free tier doesn't provide
        real-time bid/ask quotes.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with spread estimate:
            - mid_price: Estimated mid price
            - spread_estimate: Estimated spread (high - low)
            - spread_pct: Spread as percentage of mid price
        """
        quote = self.get_quote(ticker)

        if not quote:
            return None

        high = quote['high']
        low = quote['low']
        current = quote['current_price']

        # Use current price as mid, estimate spread from day's range
        mid_price = current
        spread_estimate = high - low if high > low else 0
        spread_pct = (spread_estimate / mid_price * 100) if mid_price > 0 else 0

        return {
            'mid_price': mid_price,
            'spread_estimate': spread_estimate,
            'spread_pct': spread_pct
        }
