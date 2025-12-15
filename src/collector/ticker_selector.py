"""Ticker selection module for identifying high-volume and high-gain stocks."""

from typing import List, Optional, Set
from datetime import datetime
from dataclasses import dataclass

from loguru import logger

from config.settings import settings
from src.collector.polygon_client import get_polygon_client, PolygonClientWrapper


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
        vwap: Volume weighted average price
    """
    ticker: str
    volume: float
    change_percent: float
    price: float
    prev_close: float
    vwap: Optional[float] = None


class TickerSelector:
    """
    Select target tickers based on volume and price movement.

    This class identifies high-volatility stocks by combining:
    - Top N stocks by trading volume
    - Top N stocks by percentage gain

    The union of these sets provides the target list for prediction.
    """

    def __init__(
        self,
        client: Optional[PolygonClientWrapper] = None,
        top_n_volume: Optional[int] = None,
        top_n_gainers: Optional[int] = None,
        min_price: float = 1.0,
        min_volume: float = 100000
    ):
        """
        Initialize ticker selector.

        Args:
            client: Polygon client wrapper (uses global if not provided)
            top_n_volume: Number of top volume stocks to select
            top_n_gainers: Number of top gainers to select
            min_price: Minimum stock price filter
            min_volume: Minimum daily volume filter
        """
        self.client = client or get_polygon_client()
        self.top_n_volume = top_n_volume or settings.TOP_N_VOLUME
        self.top_n_gainers = top_n_gainers or settings.TOP_N_GAINERS
        self.min_price = min_price
        self.min_volume = min_volume

        logger.info(
            f"TickerSelector initialized: top_volume={self.top_n_volume}, "
            f"top_gainers={self.top_n_gainers}, min_price=${min_price}, "
            f"min_volume={min_volume:,.0f}"
        )

    def get_all_snapshots(self) -> List[TickerMetrics]:
        """
        Get snapshots for all available stocks.

        Returns:
            List of TickerMetrics objects with current market data
        """
        try:
            logger.info("Fetching all stock snapshots from Polygon...")
            snapshots = self.client.get_snapshot_all("stocks")

            if not snapshots:
                logger.warning("No snapshots returned from API")
                return []

            ticker_metrics = []

            for snapshot in snapshots:
                try:
                    # Skip if missing essential data
                    if not hasattr(snapshot, 'ticker') or not hasattr(snapshot, 'day'):
                        continue

                    # Extract metrics
                    ticker = snapshot.ticker
                    day_data = snapshot.day
                    prev_day = snapshot.prevDay if hasattr(snapshot, 'prevDay') else None

                    # Get current price (use close or last trade price)
                    price = getattr(day_data, 'c', None) or getattr(day_data, 'close', None)
                    if not price and hasattr(snapshot, 'last_trade'):
                        price = getattr(snapshot.last_trade, 'p', None)

                    # Get volume
                    volume = getattr(day_data, 'v', None) or getattr(day_data, 'volume', None) or 0

                    # Get previous close
                    prev_close = getattr(prev_day, 'c', None) if prev_day else None
                    if not prev_close and hasattr(day_data, 'o'):
                        prev_close = day_data.o  # Use today's open as fallback

                    # Calculate change percent
                    if price and prev_close and prev_close > 0:
                        change_percent = ((price - prev_close) / prev_close) * 100
                    else:
                        # Try to get from snapshot directly
                        change_percent = getattr(snapshot, 'todaysChangePerc', 0) or 0

                    # Get VWAP if available
                    vwap = getattr(day_data, 'vw', None)

                    # Apply filters
                    if price is None or price < self.min_price:
                        continue
                    if volume < self.min_volume:
                        continue
                    if prev_close is None:
                        continue

                    # Create metrics object
                    metrics = TickerMetrics(
                        ticker=ticker,
                        volume=float(volume),
                        change_percent=float(change_percent),
                        price=float(price),
                        prev_close=float(prev_close),
                        vwap=float(vwap) if vwap else None
                    )
                    ticker_metrics.append(metrics)

                except Exception as e:
                    logger.debug(f"Failed to process snapshot for ticker: {str(e)}")
                    continue

            logger.info(f"Processed {len(ticker_metrics)} valid tickers from snapshots")
            return ticker_metrics

        except Exception as e:
            logger.error(f"Failed to get snapshots: {str(e)}")
            return []

    def get_top_by_volume(self, metrics: List[TickerMetrics]) -> List[str]:
        """
        Get top N tickers by trading volume.

        Args:
            metrics: List of ticker metrics

        Returns:
            List of ticker symbols sorted by volume (descending)
        """
        sorted_metrics = sorted(metrics, key=lambda x: x.volume, reverse=True)
        top_tickers = [m.ticker for m in sorted_metrics[:self.top_n_volume]]

        logger.info(
            f"Top {len(top_tickers)} volume tickers selected. "
            f"Range: {sorted_metrics[0].volume:,.0f} - {sorted_metrics[self.top_n_volume-1].volume:,.0f}"
            if sorted_metrics else "No tickers available"
        )

        return top_tickers

    def get_top_by_gain(self, metrics: List[TickerMetrics]) -> List[str]:
        """
        Get top N tickers by percentage gain.

        Args:
            metrics: List of ticker metrics

        Returns:
            List of ticker symbols sorted by gain (descending)
        """
        sorted_metrics = sorted(metrics, key=lambda x: x.change_percent, reverse=True)
        top_tickers = [m.ticker for m in sorted_metrics[:self.top_n_gainers]]

        logger.info(
            f"Top {len(top_tickers)} gainer tickers selected. "
            f"Range: {sorted_metrics[0].change_percent:.2f}% - "
            f"{sorted_metrics[self.top_n_gainers-1].change_percent:.2f}%"
            if sorted_metrics else "No tickers available"
        )

        return top_tickers

    def get_top_by_loss(self, metrics: List[TickerMetrics]) -> List[str]:
        """
        Get top N tickers by percentage loss (for potential reversal plays).

        Args:
            metrics: List of ticker metrics

        Returns:
            List of ticker symbols sorted by loss (ascending)
        """
        sorted_metrics = sorted(metrics, key=lambda x: x.change_percent)
        top_tickers = [m.ticker for m in sorted_metrics[:self.top_n_gainers]]

        logger.info(
            f"Top {len(top_tickers)} loser tickers selected. "
            f"Range: {sorted_metrics[0].change_percent:.2f}% - "
            f"{sorted_metrics[self.top_n_gainers-1].change_percent:.2f}%"
            if sorted_metrics else "No tickers available"
        )

        return top_tickers

    def get_target_tickers(self, include_losers: bool = False) -> List[str]:
        """
        Get union of top volume and top gainer tickers.

        This is the main method for selecting prediction targets.

        Args:
            include_losers: If True, also include top losers

        Returns:
            Unique list of ticker symbols (union of volume and gainers)
        """
        try:
            logger.info("Starting target ticker selection...")

            # Get all snapshots
            all_metrics = self.get_all_snapshots()

            if not all_metrics:
                logger.warning("No metrics available, returning empty list")
                return []

            # Get top tickers by different criteria
            volume_tickers = set(self.get_top_by_volume(all_metrics))
            gainer_tickers = set(self.get_top_by_gain(all_metrics))

            # Combine sets
            target_set: Set[str] = volume_tickers.union(gainer_tickers)

            # Optionally include losers
            if include_losers:
                loser_tickers = set(self.get_top_by_loss(all_metrics))
                target_set = target_set.union(loser_tickers)

            # Convert to sorted list
            target_tickers = sorted(list(target_set))

            # Log statistics
            overlap = volume_tickers.intersection(gainer_tickers)
            logger.info(
                f"Target ticker selection complete:\n"
                f"  - Volume tickers: {len(volume_tickers)}\n"
                f"  - Gainer tickers: {len(gainer_tickers)}\n"
                f"  - Overlap: {len(overlap)}\n"
                f"  - Total unique: {len(target_tickers)}"
            )

            return target_tickers

        except Exception as e:
            logger.error(f"Failed to get target tickers: {str(e)}")
            return []

    def get_ticker_metrics(self, ticker: str) -> Optional[TickerMetrics]:
        """
        Get current metrics for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            TickerMetrics object or None if unavailable
        """
        try:
            snapshot = self.client.get_snapshot_ticker("stocks", ticker)

            if not snapshot:
                return None

            # Extract data similar to get_all_snapshots
            day_data = snapshot.day
            prev_day = snapshot.prevDay if hasattr(snapshot, 'prevDay') else None

            price = getattr(day_data, 'c', None) or getattr(day_data, 'close', None)
            volume = getattr(day_data, 'v', None) or getattr(day_data, 'volume', None) or 0
            prev_close = getattr(prev_day, 'c', None) if prev_day else None
            vwap = getattr(day_data, 'vw', None)

            if price and prev_close and prev_close > 0:
                change_percent = ((price - prev_close) / prev_close) * 100
            else:
                change_percent = getattr(snapshot, 'todaysChangePerc', 0) or 0

            return TickerMetrics(
                ticker=ticker,
                volume=float(volume),
                change_percent=float(change_percent),
                price=float(price) if price else 0.0,
                prev_close=float(prev_close) if prev_close else 0.0,
                vwap=float(vwap) if vwap else None
            )

        except Exception as e:
            logger.error(f"Failed to get metrics for {ticker}: {str(e)}")
            return None

    def filter_tickers_by_criteria(
        self,
        tickers: List[str],
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_volume: Optional[float] = None,
        min_change_percent: Optional[float] = None
    ) -> List[str]:
        """
        Filter a list of tickers by custom criteria.

        Args:
            tickers: List of ticker symbols
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter
            min_change_percent: Minimum percentage change filter

        Returns:
            Filtered list of tickers
        """
        filtered = []

        for ticker in tickers:
            metrics = self.get_ticker_metrics(ticker)
            if not metrics:
                continue

            # Apply filters
            if min_price and metrics.price < min_price:
                continue
            if max_price and metrics.price > max_price:
                continue
            if min_volume and metrics.volume < min_volume:
                continue
            if min_change_percent and metrics.change_percent < min_change_percent:
                continue

            filtered.append(ticker)

        logger.info(f"Filtered {len(tickers)} tickers to {len(filtered)} based on criteria")
        return filtered
