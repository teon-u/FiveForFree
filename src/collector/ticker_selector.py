"""Ticker selection using Finnhub and Yahoo Finance."""

from typing import List, Optional
from dataclasses import dataclass
import yfinance as yf
from loguru import logger
from config.settings import settings
from src.collector.finnhub_client import get_finnhub_client


@dataclass
class TickerMetrics:
    """
    Metrics for a single ticker.

    Attributes:
        ticker: Stock ticker symbol
        volume: Current day volume
        change_percent: Today's percentage change
        price: Current price
        prev_close: Previous close price
        market_cap: Market capitalization
    """
    ticker: str
    volume: float
    change_percent: float
    price: float
    prev_close: float
    market_cap: Optional[float] = None


class TickerSelector:
    """
    Select target tickers using Finnhub and predefined popular stocks.

    Since Finnhub free tier doesn't provide screener functionality,
    we use a combination of:
    1. Predefined list of popular NASDAQ stocks
    2. Yahoo Finance for volume/gainer data (supplementary)
    """

    # Predefined popular NASDAQ stocks (high volume, high volatility)
    POPULAR_TICKERS = [
        # Mega cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
        # Large cap tech
        "AMD", "INTC", "CSCO", "ADBE", "AVGO", "TXN", "QCOM", "COST",
        "CMCSA", "PEP", "TMUS", "AMGN", "SBUX", "PYPL", "ABNB", "MRNA",
        # Mid cap growth
        "BKNG", "ADP", "GILD", "ADI", "REGN", "ISRG", "PANW", "MU",
        "LRCX", "KLAC", "SNPS", "CDNS", "ASML", "MELI", "NXPI", "MAR",
        # High volatility
        "ORLY", "CTAS", "DXCM", "WDAY", "IDXX", "MNST", "VRTX", "LULU",
        "CHTR", "MCHP", "FTNT", "PAYX", "ADSK", "ROST", "FAST", "ODFL",
        # Growth stocks
        "CPRT", "VRSK", "EA", "PCAR", "KDP", "CTSH", "ANSS", "DLTR",
        "CSGP", "ILMN", "EBAY", "WBA", "BIIB", "ZS", "DDOG", "TEAM",
        # High beta
        "CRWD", "OKTA", "ZM", "DOCU", "SPLK", "MDB", "NET", "SNOW",
        "ESTC", "BILL", "RIVN", "LCID", "COIN", "RBLX", "U", "DASH",
        "HOOD", "SOFI", "PLTR", "UPST", "AFRM", "SQ", "UBER", "LYFT"
    ]

    def __init__(
        self,
        top_n_volume: Optional[int] = None,
        top_n_gainers: Optional[int] = None,
        min_price: float = 5.0,
        min_volume: float = 1_000_000
    ):
        """
        Initialize ticker selector.

        Args:
            top_n_volume: Number of tickers to return (default from settings)
            top_n_gainers: Number of gainers to include (default from settings)
            min_price: Minimum stock price filter
            min_volume: Minimum daily volume filter
        """
        self.finnhub_client = get_finnhub_client()
        self.top_n_volume = top_n_volume or settings.TOP_N_VOLUME
        self.top_n_gainers = top_n_gainers or settings.TOP_N_GAINERS
        self.min_price = min_price
        self.min_volume = min_volume

        logger.info(
            f"TickerSelector initialized: top_volume={self.top_n_volume}, "
            f"top_gainers={self.top_n_gainers}, min_price=${min_price}, "
            f"min_volume={min_volume:,.0f}"
        )

    def get_target_tickers(self) -> List[str]:
        """
        Get target ticker list.

        Strategy:
        1. Start with predefined popular NASDAQ stocks
        2. Filter by price and volume using Finnhub quotes
        3. Sort by volume to get top N

        Returns:
            List of ticker symbols
        """
        logger.info("Selecting target tickers...")

        valid_tickers = []
        ticker_data = []

        # Get quotes for predefined tickers
        for ticker in self.POPULAR_TICKERS[:100]:  # Limit to avoid API quota
            try:
                quote = self.finnhub_client.get_quote(ticker)

                # Skip if no data
                if quote.get('c') is None or quote.get('c') == 0:
                    continue

                price = quote['c']
                prev_close = quote.get('pc', price)

                # Apply filters
                if price < self.min_price:
                    continue

                # Get additional data from yfinance for volume
                try:
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    volume = info.get('volume', 0)
                    market_cap = info.get('marketCap', 0)

                    if volume < self.min_volume:
                        continue

                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0

                    ticker_data.append({
                        'ticker': ticker,
                        'volume': volume,
                        'price': price,
                        'change_pct': change_pct,
                        'market_cap': market_cap
                    })

                except Exception as e:
                    logger.debug(f"Could not get yfinance data for {ticker}: {e}")
                    # Use Finnhub data only
                    ticker_data.append({
                        'ticker': ticker,
                        'volume': 0,  # Unknown
                        'price': price,
                        'change_pct': 0,
                        'market_cap': 0
                    })

            except Exception as e:
                logger.debug(f"Error fetching quote for {ticker}: {e}")
                continue

        # Sort by volume (descending)
        ticker_data.sort(key=lambda x: x['volume'], reverse=True)

        # Get top N by volume
        top_volume = ticker_data[:self.top_n_volume]

        # Get top N gainers
        gainers = sorted(ticker_data, key=lambda x: x['change_pct'], reverse=True)
        top_gainers = gainers[:self.top_n_gainers]

        # Combine (union)
        result_tickers = list(set([t['ticker'] for t in top_volume] +
                                  [t['ticker'] for t in top_gainers]))

        logger.info(
            f"Selected {len(result_tickers)} tickers "
            f"(top volume: {len(top_volume)}, top gainers: {len(top_gainers)})"
        )

        return result_tickers

    def get_ticker_metrics(self, ticker: str) -> Optional[TickerMetrics]:
        """
        Get detailed metrics for a single ticker.

        Args:
            ticker: Stock symbol

        Returns:
            TickerMetrics object or None if data unavailable
        """
        try:
            # Get quote from Finnhub
            quote = self.finnhub_client.get_quote(ticker)

            if quote.get('c') is None:
                return None

            price = quote['c']
            prev_close = quote.get('pc', price)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0

            # Try to get volume from yfinance
            volume = 0
            market_cap = None
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                volume = info.get('volume', 0)
                market_cap = info.get('marketCap')
            except Exception as e:
                logger.debug(f"Could not get yfinance data for {ticker}: {e}")

            return TickerMetrics(
                ticker=ticker,
                volume=volume,
                change_percent=change_pct,
                price=price,
                prev_close=prev_close,
                market_cap=market_cap
            )

        except Exception as e:
            logger.error(f"Error getting metrics for {ticker}: {e}")
            return None

    def get_all_metrics(self, tickers: List[str]) -> List[TickerMetrics]:
        """
        Get metrics for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of TickerMetrics objects
        """
        metrics = []
        for ticker in tickers:
            metric = self.get_ticker_metrics(ticker)
            if metric:
                metrics.append(metric)

        logger.info(f"Retrieved metrics for {len(metrics)}/{len(tickers)} tickers")
        return metrics
