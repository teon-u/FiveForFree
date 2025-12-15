"""Data collection modules using Yahoo Finance and Finnhub."""

# Yahoo Finance for price data (primary)
from src.collector.minute_bars import (
    MinuteBarCollector,
    MinuteBar,
)

# Finnhub for real-time quotes and company info (secondary)
from src.collector.finnhub_client import (
    get_finnhub_client,
    FinnhubClientWrapper,
    RateLimitError,
)

# Ticker selection (uses both yfinance and Finnhub)
from src.collector.ticker_selector import (
    TickerSelector,
    TickerMetrics,
)

# Simplified quotes (Finnhub real-time only)
from src.collector.quotes import (
    QuoteCollector,
)

# Market context (Finnhub quotes)
from src.collector.market_context import (
    MarketContextCollector,
    MarketContext,
    MarketIndicator,
    SectorPerformance,
    SectorETF,
)

__all__ = [
    # Minute bars (Yahoo Finance)
    'MinuteBarCollector',
    'MinuteBar',
    # Finnhub client
    'get_finnhub_client',
    'FinnhubClientWrapper',
    'RateLimitError',
    # Ticker selection
    'TickerSelector',
    'TickerMetrics',
    # Quotes
    'QuoteCollector',
    # Market context
    'MarketContextCollector',
    'MarketContext',
    'MarketIndicator',
    'SectorPerformance',
    'SectorETF',
]
