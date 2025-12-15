"""Data collection modules for Finnhub API."""

from src.collector.finnhub_client import (
    FinnhubClientWrapper,
    get_finnhub_client,
    RateLimitError
)
from src.collector.ticker_selector import (
    TickerSelector,
    TickerMetrics
)
from src.collector.minute_bars import (
    MinuteBarCollector,
    MinuteBar
)
from src.collector.quotes import (
    QuoteCollector
)
from src.collector.market_context import (
    MarketContextCollector,
    MarketContext,
    MarketIndicator,
    SectorPerformance,
    SectorETF
)

__all__ = [
    # Finnhub client
    'FinnhubClientWrapper',
    'get_finnhub_client',
    'RateLimitError',

    # Ticker selection
    'TickerSelector',
    'TickerMetrics',

    # Minute bars
    'MinuteBarCollector',
    'MinuteBar',

    # Quotes
    'QuoteCollector',

    # Market context
    'MarketContextCollector',
    'MarketContext',
    'MarketIndicator',
    'SectorPerformance',
    'SectorETF',
]
