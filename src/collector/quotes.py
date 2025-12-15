"""Level 2 order book and quote data collection module."""

from typing import List, Optional, Tuple, Dict
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
from loguru import logger

from config.settings import settings
from src.collector.polygon_client import get_polygon_client, PolygonClientWrapper


@dataclass
class OrderBookLevel:
    """
    Represents a single price level in the order book.

    Attributes:
        price: Price level
        size: Total size (shares) at this level
    """
    price: float
    size: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OrderBookSnapshot:
    """
    Level 2 order book snapshot data.

    Attributes:
        ticker: Stock ticker symbol
        timestamp: Snapshot timestamp
        bids: List of bid levels (price, size)
        asks: List of ask levels (price, size)
        bid_total_volume: Total volume on bid side
        ask_total_volume: Total volume on ask side
        imbalance: Order flow imbalance (-1 to 1)
        spread: Bid-ask spread
        mid_price: Mid-point price
        depth_weighted_mid: Depth-weighted mid price
    """
    ticker: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    bid_total_volume: float
    ask_total_volume: float
    imbalance: float
    spread: float
    mid_price: float
    depth_weighted_mid: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert OrderBookLevel objects to tuples for easier serialization
        data['bids'] = [(b.price, b.size) for b in self.bids]
        data['asks'] = [(a.price, a.size) for a in self.asks]
        return data


@dataclass
class Quote:
    """
    Represents a single quote (BBO - Best Bid and Offer).

    Attributes:
        ticker: Stock ticker symbol
        timestamp: Quote timestamp
        bid_price: Best bid price
        bid_size: Best bid size
        ask_price: Best ask price
        ask_size: Best ask size
        spread: Bid-ask spread
        spread_pct: Spread as percentage of mid price
    """
    ticker: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float
    spread_pct: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2


class QuoteCollector:
    """
    Collector for Level 2 order book and quote data.

    Provides methods to fetch order book snapshots and calculate
    microstructure features like imbalance and weighted mid price.
    """

    def __init__(
        self,
        client: Optional[PolygonClientWrapper] = None,
        max_levels: int = 10
    ):
        """
        Initialize quote collector.

        Args:
            client: Polygon client wrapper (uses global if not provided)
            max_levels: Maximum order book levels to capture per side
        """
        self.client = client or get_polygon_client()
        self.max_levels = max_levels
        logger.info(f"QuoteCollector initialized with max_levels={max_levels}")

    def get_order_book_snapshot(self, ticker: str) -> Optional[OrderBookSnapshot]:
        """
        Get Level 2 order book snapshot for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            OrderBookSnapshot object or None if unavailable
        """
        try:
            logger.debug(f"Fetching order book snapshot for {ticker}")

            # Get snapshot from API
            snapshot = self.client.get_snapshot_ticker("stocks", ticker)

            if not snapshot:
                logger.debug(f"No snapshot available for {ticker}")
                return None

            # Check if order book data is available
            if not hasattr(snapshot, 'book') or not snapshot.book:
                logger.debug(f"No order book data in snapshot for {ticker}")
                return None

            book = snapshot.book

            # Extract bids and asks
            bids = []
            asks = []

            # Process bids (buy orders)
            if hasattr(book, 'bids') and book.bids:
                for bid in book.bids[:self.max_levels]:
                    try:
                        price = float(getattr(bid, 'p', 0) or getattr(bid, 'price', 0))
                        size = float(getattr(bid, 'x', 0) or getattr(bid, 'size', 0))
                        if price > 0 and size > 0:
                            bids.append(OrderBookLevel(price=price, size=size))
                    except Exception as e:
                        logger.debug(f"Failed to parse bid: {str(e)}")

            # Process asks (sell orders)
            if hasattr(book, 'asks') and book.asks:
                for ask in book.asks[:self.max_levels]:
                    try:
                        price = float(getattr(ask, 'p', 0) or getattr(ask, 'price', 0))
                        size = float(getattr(ask, 'x', 0) or getattr(ask, 'size', 0))
                        if price > 0 and size > 0:
                            asks.append(OrderBookLevel(price=price, size=size))
                    except Exception as e:
                        logger.debug(f"Failed to parse ask: {str(e)}")

            # Ensure we have data
            if not bids or not asks:
                logger.debug(f"Insufficient order book data for {ticker}")
                return None

            # Calculate metrics
            bid_total_volume = sum(b.size for b in bids)
            ask_total_volume = sum(a.size for a in asks)

            # Order flow imbalance
            total_volume = bid_total_volume + ask_total_volume
            if total_volume > 0:
                imbalance = (bid_total_volume - ask_total_volume) / total_volume
            else:
                imbalance = 0.0

            # Spread
            spread = asks[0].price - bids[0].price

            # Mid price
            mid_price = (bids[0].price + asks[0].price) / 2

            # Depth-weighted mid price
            depth_weighted_mid = self._calculate_depth_weighted_mid(bids, asks)

            # Create snapshot
            snapshot_obj = OrderBookSnapshot(
                ticker=ticker,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                bid_total_volume=bid_total_volume,
                ask_total_volume=ask_total_volume,
                imbalance=imbalance,
                spread=spread,
                mid_price=mid_price,
                depth_weighted_mid=depth_weighted_mid
            )

            logger.debug(
                f"Order book snapshot for {ticker}: "
                f"imbalance={imbalance:.3f}, spread=${spread:.4f}"
            )

            return snapshot_obj

        except Exception as e:
            logger.error(f"Failed to get order book snapshot for {ticker}: {str(e)}")
            return None

    def _calculate_depth_weighted_mid(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel]
    ) -> Optional[float]:
        """
        Calculate depth-weighted mid price.

        The depth-weighted mid price gives more weight to price levels
        with higher volume, providing a better estimate of true market price.

        Args:
            bids: List of bid levels
            asks: List of ask levels

        Returns:
            Depth-weighted mid price or None
        """
        try:
            if not bids or not asks:
                return None

            # Calculate weighted bid price
            bid_weights = [b.size for b in bids]
            bid_total = sum(bid_weights)
            if bid_total == 0:
                return None

            weighted_bid = sum(b.price * b.size for b in bids) / bid_total

            # Calculate weighted ask price
            ask_weights = [a.size for a in asks]
            ask_total = sum(ask_weights)
            if ask_total == 0:
                return None

            weighted_ask = sum(a.price * a.size for a in asks) / ask_total

            # Return average
            return (weighted_bid + weighted_ask) / 2

        except Exception as e:
            logger.debug(f"Failed to calculate depth-weighted mid: {str(e)}")
            return None

    def get_last_quote(self, ticker: str) -> Optional[Quote]:
        """
        Get the last quote (BBO - Best Bid and Offer) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote object or None if unavailable
        """
        try:
            logger.debug(f"Fetching last quote for {ticker}")

            quote_data = self.client.get_last_quote(ticker)

            if not quote_data:
                return None

            # Extract quote information
            bid_price = float(getattr(quote_data, 'bid_price', 0) or getattr(quote_data, 'p', 0))
            bid_size = float(getattr(quote_data, 'bid_size', 0) or getattr(quote_data, 's', 0))
            ask_price = float(getattr(quote_data, 'ask_price', 0) or getattr(quote_data, 'P', 0))
            ask_size = float(getattr(quote_data, 'ask_size', 0) or getattr(quote_data, 'S', 0))

            if bid_price <= 0 or ask_price <= 0:
                return None

            # Calculate spread
            spread = ask_price - bid_price
            mid_price = (bid_price + ask_price) / 2
            spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0

            # Get timestamp
            timestamp = datetime.now()
            if hasattr(quote_data, 't'):
                timestamp = datetime.fromtimestamp(quote_data.t / 1e9)  # Nanoseconds to seconds

            quote = Quote(
                ticker=ticker,
                timestamp=timestamp,
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                spread=spread,
                spread_pct=spread_pct
            )

            return quote

        except Exception as e:
            logger.error(f"Failed to get last quote for {ticker}: {str(e)}")
            return None

    def get_order_book_features(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Extract microstructure features from order book.

        Returns a dictionary of features useful for prediction models.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of order book features
        """
        snapshot = self.get_order_book_snapshot(ticker)

        if not snapshot:
            return None

        features = {
            # Spread features
            'spread': snapshot.spread,
            'spread_pct': (snapshot.spread / snapshot.mid_price * 100) if snapshot.mid_price > 0 else 0,

            # Imbalance features
            'order_imbalance': snapshot.imbalance,
            'bid_volume': snapshot.bid_total_volume,
            'ask_volume': snapshot.ask_total_volume,
            'total_volume': snapshot.bid_total_volume + snapshot.ask_total_volume,

            # Price features
            'mid_price': snapshot.mid_price,
            'depth_weighted_mid': snapshot.depth_weighted_mid or snapshot.mid_price,

            # Book depth features
            'bid_levels': len(snapshot.bids),
            'ask_levels': len(snapshot.asks),
            'total_levels': len(snapshot.bids) + len(snapshot.asks),
        }

        # Add price deviation
        if snapshot.depth_weighted_mid:
            features['price_deviation_from_weighted'] = (
                (snapshot.mid_price - snapshot.depth_weighted_mid) / snapshot.mid_price * 100
            )

        # Add weighted spread
        if len(snapshot.bids) >= 3 and len(snapshot.asks) >= 3:
            features['spread_level_3'] = snapshot.asks[2].price - snapshot.bids[2].price

        return features

    def get_quotes_batch(
        self,
        tickers: List[str]
    ) -> Dict[str, Optional[Quote]]:
        """
        Get last quotes for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker to Quote object
        """
        results = {}

        logger.info(f"Fetching quotes for {len(tickers)} tickers...")

        for ticker in tickers:
            quote = self.get_last_quote(ticker)
            results[ticker] = quote

        successful = sum(1 for q in results.values() if q is not None)
        logger.info(f"Successfully fetched quotes for {successful}/{len(tickers)} tickers")

        return results

    def get_order_books_batch(
        self,
        tickers: List[str]
    ) -> Dict[str, Optional[OrderBookSnapshot]]:
        """
        Get order book snapshots for multiple tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker to OrderBookSnapshot object
        """
        results = {}

        logger.info(f"Fetching order books for {len(tickers)} tickers...")

        for ticker in tickers:
            snapshot = self.get_order_book_snapshot(ticker)
            results[ticker] = snapshot

        successful = sum(1 for s in results.values() if s is not None)
        logger.info(f"Successfully fetched order books for {successful}/{len(tickers)} tickers")

        return results

    def monitor_spread_changes(
        self,
        ticker: str,
        threshold_pct: float = 0.1
    ) -> Optional[Dict[str, any]]:
        """
        Monitor significant changes in bid-ask spread.

        Args:
            ticker: Stock ticker symbol
            threshold_pct: Threshold for significant spread change (%)

        Returns:
            Dictionary with spread change information or None
        """
        quote = self.get_last_quote(ticker)

        if not quote:
            return None

        # Check if spread is unusually wide
        is_wide = quote.spread_pct > threshold_pct

        return {
            'ticker': ticker,
            'spread': quote.spread,
            'spread_pct': quote.spread_pct,
            'is_wide': is_wide,
            'bid_price': quote.bid_price,
            'ask_price': quote.ask_price,
            'timestamp': quote.timestamp
        }

    def calculate_liquidity_score(self, ticker: str) -> Optional[float]:
        """
        Calculate a liquidity score based on order book depth and spread.

        Higher score indicates better liquidity.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Liquidity score (0-100) or None
        """
        snapshot = self.get_order_book_snapshot(ticker)

        if not snapshot:
            return None

        try:
            # Components of liquidity score
            # 1. Spread (lower is better)
            spread_score = max(0, 100 - (snapshot.spread / snapshot.mid_price * 10000))

            # 2. Volume (higher is better)
            total_volume = snapshot.bid_total_volume + snapshot.ask_total_volume
            volume_score = min(100, total_volume / 10000)  # Normalize

            # 3. Balance (closer to 0 is better)
            balance_score = max(0, 100 - abs(snapshot.imbalance) * 100)

            # 4. Depth (more levels is better)
            depth_score = min(100, (len(snapshot.bids) + len(snapshot.asks)) / self.max_levels * 50)

            # Weighted average
            liquidity_score = (
                spread_score * 0.4 +
                volume_score * 0.3 +
                balance_score * 0.2 +
                depth_score * 0.1
            )

            return round(liquidity_score, 2)

        except Exception as e:
            logger.error(f"Failed to calculate liquidity score for {ticker}: {str(e)}")
            return None
