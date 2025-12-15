"""Polygon.io API client wrapper with rate limiting and error handling."""

from typing import Optional, Any
from datetime import datetime, timedelta
import time
from functools import wraps

from polygon import RESTClient
from polygon.exceptions import BadResponse, NoResultsError
from loguru import logger

from config.settings import settings


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


class PolygonClientWrapper:
    """
    Wrapper around Polygon.io REST client with enhanced error handling and rate limiting.

    Features:
    - Automatic retry on transient errors
    - Rate limit handling
    - Comprehensive error logging
    - Connection pooling

    Attributes:
        client: Polygon REST client instance
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        rate_limit_delay: Delay when rate limited in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 60.0
    ):
        """
        Initialize Polygon client wrapper.

        Args:
            api_key: Polygon.io API key (uses settings if not provided)
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            rate_limit_delay: Delay when rate limit is hit
        """
        self.api_key = api_key or settings.POLYGON_API_KEY
        self.client = RESTClient(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay

        logger.info("Polygon client initialized successfully")

    def _retry_on_error(self, func):
        """
        Decorator to retry API calls on transient errors.

        Args:
            func: Function to wrap with retry logic

        Returns:
            Wrapped function with retry capability
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Request succeeded on attempt {attempt + 1}")
                    return result

                except BadResponse as e:
                    last_exception = e
                    # Check if rate limited (HTTP 429)
                    if hasattr(e, 'status') and e.status == 429:
                        logger.warning(f"Rate limited. Waiting {self.rate_limit_delay}s...")
                        time.sleep(self.rate_limit_delay)
                        continue

                    # Retry on server errors (5xx)
                    if hasattr(e, 'status') and 500 <= e.status < 600:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Server error {e.status}. Retry {attempt + 1}/{self.max_retries} "
                            f"after {delay}s"
                        )
                        time.sleep(delay)
                        continue

                    # Don't retry client errors (4xx except 429)
                    logger.error(f"Client error {e.status}: {str(e)}")
                    raise

                except NoResultsError:
                    logger.debug("No results found for query")
                    return None

                except Exception as e:
                    last_exception = e
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Request failed: {str(e)}. Retry {attempt + 1}/{self.max_retries} "
                        f"after {delay}s"
                    )
                    time.sleep(delay)

            # All retries exhausted
            logger.error(f"All {self.max_retries} retries exhausted. Last error: {str(last_exception)}")
            raise last_exception

        return wrapper

    @property
    def get_snapshot_all(self):
        """
        Get all tickers snapshot with retry logic.

        Returns:
            Wrapped snapshot function
        """
        return self._retry_on_error(self.client.get_snapshot_all)

    @property
    def get_snapshot_ticker(self):
        """
        Get single ticker snapshot with retry logic.

        Returns:
            Wrapped snapshot function
        """
        return self._retry_on_error(self.client.get_snapshot_ticker)

    @property
    def get_aggs(self):
        """
        Get aggregates (bars) with retry logic.

        Returns:
            Wrapped aggregates function
        """
        return self._retry_on_error(self.client.get_aggs)

    @property
    def list_aggs(self):
        """
        List aggregates (bars) with retry logic.

        Returns:
            Wrapped list aggregates function
        """
        return self._retry_on_error(self.client.list_aggs)

    def get_minute_bars(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        limit: int = 50000
    ) -> Optional[list]:
        """
        Get minute-level OHLCV bars for a ticker.

        Args:
            ticker: Stock ticker symbol
            from_date: Start date/time
            to_date: End date/time
            limit: Maximum number of bars to return

        Returns:
            List of aggregate bars or None if no data
        """
        try:
            bars = list(self.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="minute",
                from_=from_date.strftime("%Y-%m-%d"),
                to=to_date.strftime("%Y-%m-%d"),
                limit=limit
            ))

            logger.debug(f"Retrieved {len(bars)} minute bars for {ticker}")
            return bars if bars else None

        except Exception as e:
            logger.error(f"Failed to get minute bars for {ticker}: {str(e)}")
            return None

    def get_previous_close(self, ticker: str) -> Optional[dict]:
        """
        Get previous day's close price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Previous close data or None
        """
        try:
            @self._retry_on_error
            def _get_prev_close():
                return self.client.get_previous_close(ticker)

            result = _get_prev_close()
            if result and len(result) > 0:
                return result[0]
            return None

        except Exception as e:
            logger.error(f"Failed to get previous close for {ticker}: {str(e)}")
            return None

    def get_last_trade(self, ticker: str) -> Optional[dict]:
        """
        Get last trade for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Last trade data or None
        """
        try:
            @self._retry_on_error
            def _get_last_trade():
                return self.client.get_last_trade(ticker)

            return _get_last_trade()

        except Exception as e:
            logger.error(f"Failed to get last trade for {ticker}: {str(e)}")
            return None

    def get_last_quote(self, ticker: str) -> Optional[dict]:
        """
        Get last quote (bid/ask) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Last quote data or None
        """
        try:
            @self._retry_on_error
            def _get_last_quote():
                return self.client.get_last_quote(ticker)

            return _get_last_quote()

        except Exception as e:
            logger.error(f"Failed to get last quote for {ticker}: {str(e)}")
            return None

    def health_check(self) -> bool:
        """
        Check if API is accessible and credentials are valid.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get a simple snapshot to verify connection
            snapshot = self.get_snapshot_ticker("stocks", "AAPL")
            if snapshot:
                logger.info("Polygon API health check: OK")
                return True
            return False

        except Exception as e:
            logger.error(f"Polygon API health check failed: {str(e)}")
            return False


# Global client instance
_client_instance: Optional[PolygonClientWrapper] = None


def get_polygon_client() -> PolygonClientWrapper:
    """
    Get or create global Polygon client instance (singleton pattern).

    Returns:
        PolygonClientWrapper instance
    """
    global _client_instance

    if _client_instance is None:
        _client_instance = PolygonClientWrapper()
        logger.info("Created global Polygon client instance")

    return _client_instance
