"""Finnhub API client wrapper with rate limiting."""
import time
import finnhub
from loguru import logger
from config.settings import settings


class RateLimitError(Exception):
    """Rate limit exceeded exception."""
    pass


class FinnhubClientWrapper:
    """
    Wrapper for Finnhub API with rate limiting.

    Finnhub Free Tier:
    - 60 API calls per minute
    - 1 year of historical data
    - No Level 2 order book data
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Finnhub client.

        Args:
            api_key: Finnhub API key (defaults to settings)
        """
        self.api_key = api_key or settings.FINNHUB_API_KEY
        self.client = finnhub.Client(api_key=self.api_key)
        self.last_call_time = 0
        self.min_call_interval = settings.API_CALL_DELAY  # 1 second
        self.call_count = 0

        logger.info("Finnhub client initialized")

    def _rate_limit(self):
        """
        Rate limiting: 60 calls/minute = 1 call/second.
        Ensures we don't exceed API limits.
        """
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            sleep_time = self.min_call_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_call_time = time.time()
        self.call_count += 1

    def get_candles(self, symbol: str, resolution: str, from_ts: int, to_ts: int) -> dict:
        """
        Get candlestick data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            resolution: 1, 5, 15, 30, 60, D, W, M
            from_ts: Unix timestamp (from)
            to_ts: Unix timestamp (to)

        Returns:
            Dict with keys: c, h, l, o, t, v, s

        Note:
            Free tier: Recommend 5-minute resolution for stability
        """
        self._rate_limit()

        try:
            result = self.client.stock_candles(symbol, resolution, from_ts, to_ts)
            logger.debug(f"Fetched candles for {symbol}: status={result.get('s')}")
            return result
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            raise

    def get_quote(self, symbol: str) -> dict:
        """
        Get real-time quote.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with keys: c (current), h (high), l (low), o (open),
            pc (previous close), t (timestamp)
        """
        self._rate_limit()

        try:
            quote = self.client.quote(symbol)
            logger.debug(f"Fetched quote for {symbol}: price=${quote.get('c')}")
            return quote
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    def get_company_profile(self, symbol: str) -> dict:
        """
        Get company profile.

        Args:
            symbol: Stock symbol

        Returns:
            Company information (name, industry, market cap, etc.)
        """
        self._rate_limit()

        try:
            profile = self.client.company_profile2(symbol=symbol)
            logger.debug(f"Fetched profile for {symbol}: {profile.get('name')}")
            return profile
        except Exception as e:
            logger.error(f"Error fetching profile for {symbol}: {e}")
            raise

    def get_market_news(self, category: str = "general", min_id: int = 0) -> list:
        """
        Get market news.

        Args:
            category: general, forex, crypto, merger
            min_id: Minimum news ID

        Returns:
            List of news articles
        """
        self._rate_limit()

        try:
            news = self.client.general_news(category, min_id=min_id)
            logger.debug(f"Fetched {len(news)} news articles")
            return news
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check API connectivity.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple check: get quote for SPY
            result = self.get_quote("SPY")
            return result.get('c') is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_call_count(self) -> int:
        """Get total API calls made."""
        return self.call_count

    def reset_call_count(self):
        """Reset call counter."""
        self.call_count = 0
        logger.info("Call counter reset")


# Global singleton instance
_client = None


def get_finnhub_client() -> FinnhubClientWrapper:
    """
    Get global Finnhub client instance (singleton).

    Returns:
        FinnhubClientWrapper instance
    """
    global _client
    if _client is None:
        _client = FinnhubClientWrapper()
    return _client
