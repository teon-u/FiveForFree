"""Data collection modules for Polygon.io API."""

from src.collector.polygon_client import (
    PolygonClientWrapper,
    get_polygon_client,
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
    QuoteCollector,
    OrderBookSnapshot,
    OrderBookLevel,
    Quote
)
from src.collector.market_context import (
    MarketContextCollector,
    MarketContext,
    MarketIndicator,
    SectorPerformance,
    SectorETF
)

__all__ = [
    # Polygon client
    'PolygonClientWrapper',
    'get_polygon_client',
    'RateLimitError',

    # Ticker selection
    'TickerSelector',
    'TickerMetrics',

    # Minute bars
    'MinuteBarCollector',
    'MinuteBar',

    # Quotes and order book
    'QuoteCollector',
    'OrderBookSnapshot',
    'OrderBookLevel',
    'Quote',

    # Market context
    'MarketContextCollector',
    'MarketContext',
    'MarketIndicator',
    'SectorPerformance',
    'SectorETF',
]
