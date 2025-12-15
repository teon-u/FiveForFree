"""Ticker selection using real-time volume and gainers from Yahoo Finance."""

from typing import List, Optional, Dict
from dataclasses import dataclass
import yfinance as yf
import pandas as pd
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
        category: 'volume' or 'gainer'
    """
    ticker: str
    volume: float
    change_percent: float
    price: float
    prev_close: float
    market_cap: Optional[float] = None
    category: str = 'volume'  # 'volume' or 'gainer'


class TickerSelector:
    """
    Select target tickers using real-time volume and gainers from Yahoo Finance.

    Returns two separate lists:
    1. Top N by volume
    2. Top N by gainers (% change)

    This allows UI to display them separately with toggle buttons.
    """

    # NASDAQ 100 + high volume stocks as screening universe
    NASDAQ_UNIVERSE = [
        # Mega cap
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AVGO",
        # Large cap tech
        "AMD", "INTC", "CSCO", "ADBE", "ORCL", "TXN", "QCOM", "COST", "CMCSA", "PEP",
        "TMUS", "AMGN", "SBUX", "PYPL", "ABNB", "MRNA", "BKNG", "ADP", "GILD", "ADI",
        # Mid/Small cap growth
        "REGN", "ISRG", "PANW", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ASML", "MELI",
        "NXPI", "MAR", "ORLY", "CTAS", "DXCM", "WDAY", "IDXX", "MNST", "VRTX", "LULU",
        "CHTR", "MCHP", "FTNT", "PAYX", "ADSK", "ROST", "FAST", "ODFL", "CPRT", "VRSK",
        "EA", "PCAR", "KDP", "CTSH", "ANSS", "DLTR", "CSGP", "ILMN", "EBAY", "WBA",
        # High volatility
        "BIIB", "ZS", "DDOG", "TEAM", "CRWD", "OKTA", "ZM", "DOCU", "SPLK", "MDB",
        "NET", "SNOW", "ESTC", "BILL", "RIVN", "LCID", "COIN", "RBLX", "U", "DASH",
        "HOOD", "SOFI", "PLTR", "UPST", "AFRM", "SQ", "UBER", "LYFT", "ABNB", "SPOT",
        # Additional high volume
        "AMAT", "NTES", "MRVL", "BIDU", "JD", "PDD", "NDAQ", "ALGN", "SIRI", "GEHC",
        "ON", "TTWO", "WBD", "ENPH", "FANG", "SGEN", "VRSN", "ICLR", "BMRN", "SWKS"
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
            top_n_volume: Number of top volume tickers (default from settings)
            top_n_gainers: Number of top gainers (default from settings)
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
            f"top_gainers={self.top_n_gainers}"
        )

    def get_realtime_metrics(self, ticker: str) -> Optional[TickerMetrics]:
        """
        Get real-time metrics for a single ticker.

        Args:
            ticker: Stock symbol

        Returns:
            TickerMetrics or None if data unavailable
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info

            # Get current price and volume
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            volume = info.get('volume') or info.get('regularMarketVolume', 0)
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
            market_cap = info.get('marketCap')

            if not price or not prev_close:
                return None

            # Calculate change percent
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0

            return TickerMetrics(
                ticker=ticker,
                volume=volume,
                change_percent=change_pct,
                price=price,
                prev_close=prev_close,
                market_cap=market_cap
            )

        except Exception as e:
            logger.debug(f"Error getting metrics for {ticker}: {e}")
            return None

    def get_top_by_volume(self) -> List[TickerMetrics]:
        """
        Get top N tickers by trading volume.

        Returns:
            List of TickerMetrics sorted by volume (descending)
        """
        logger.info("Fetching top tickers by volume...")

        metrics_list = []

        for ticker in self.NASDAQ_UNIVERSE:
            metrics = self.get_realtime_metrics(ticker)

            if not metrics:
                continue

            # Apply filters
            if metrics.price < self.min_price:
                continue
            if metrics.volume < self.min_volume:
                continue

            metrics.category = 'volume'
            metrics_list.append(metrics)

        # Sort by volume
        metrics_list.sort(key=lambda x: x.volume, reverse=True)

        # Get top N
        top_volume = metrics_list[:self.top_n_volume]

        logger.info(f"Selected {len(top_volume)} top volume tickers")

        return top_volume

    def get_top_by_gainers(self) -> List[TickerMetrics]:
        """
        Get top N tickers by percentage gain.

        Returns:
            List of TickerMetrics sorted by change_percent (descending)
        """
        logger.info("Fetching top gainers...")

        metrics_list = []

        for ticker in self.NASDAQ_UNIVERSE:
            metrics = self.get_realtime_metrics(ticker)

            if not metrics:
                continue

            # Apply filters
            if metrics.price < self.min_price:
                continue
            if metrics.volume < self.min_volume:
                continue

            metrics.category = 'gainer'
            metrics_list.append(metrics)

        # Sort by change percent
        metrics_list.sort(key=lambda x: x.change_percent, reverse=True)

        # Get top N
        top_gainers = metrics_list[:self.top_n_gainers]

        logger.info(f"Selected {len(top_gainers)} top gainers")

        return top_gainers

    def get_both_categories(self) -> Dict[str, List[TickerMetrics]]:
        """
        Get both top volume and top gainers.

        Returns:
            Dict with 'volume' and 'gainers' keys, each containing list of TickerMetrics
        """
        logger.info("Fetching both volume and gainers...")

        return {
            'volume': self.get_top_by_volume(),
            'gainers': self.get_top_by_gainers()
        }

    def get_all_unique_tickers(self) -> List[str]:
        """
        Get all unique tickers from both categories (for data collection).

        Returns:
            List of unique ticker symbols
        """
        categories = self.get_both_categories()

        all_tickers = set()
        for metrics_list in categories.values():
            all_tickers.update(m.ticker for m in metrics_list)

        logger.info(f"Total unique tickers: {len(all_tickers)}")

        return list(all_tickers)

    # Legacy method for backward compatibility
    def get_target_tickers(self) -> List[str]:
        """
        Get target ticker list (all unique tickers from both categories).

        Note: For UI separation, use get_both_categories() instead.

        Returns:
            List of ticker symbols
        """
        return self.get_all_unique_tickers()

    def get_ticker_metrics(self, ticker: str) -> Optional[TickerMetrics]:
        """
        Get detailed metrics for a single ticker.

        Args:
            ticker: Stock symbol

        Returns:
            TickerMetrics object or None if data unavailable
        """
        return self.get_realtime_metrics(ticker)

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
            metric = self.get_realtime_metrics(ticker)
            if metric:
                metrics.append(metric)

        logger.info(f"Retrieved metrics for {len(metrics)}/{len(tickers)} tickers")
        return metrics
