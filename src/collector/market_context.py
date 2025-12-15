"""Market context data collection module for SPY, QQQ, VIX, and sector ETFs."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from loguru import logger

from config.settings import settings
from src.collector.polygon_client import get_polygon_client, PolygonClientWrapper


class SectorETF(Enum):
    """Major sector ETFs for market context."""
    # SPDR Sector ETFs
    XLK = "XLK"  # Technology
    XLF = "XLF"  # Financials
    XLV = "XLV"  # Healthcare
    XLE = "XLE"  # Energy
    XLY = "XLY"  # Consumer Discretionary
    XLP = "XLP"  # Consumer Staples
    XLI = "XLI"  # Industrials
    XLB = "XLB"  # Materials
    XLU = "XLU"  # Utilities
    XLRE = "XLRE"  # Real Estate
    XLC = "XLC"  # Communication Services


@dataclass
class MarketIndicator:
    """
    Market-wide indicator data.

    Attributes:
        symbol: Ticker symbol (SPY, QQQ, VIX, etc.)
        timestamp: Data timestamp
        price: Current price
        change_percent: Percentage change from previous close
        volume: Trading volume
        prev_close: Previous close price
    """
    symbol: str
    timestamp: datetime
    price: float
    change_percent: float
    volume: float
    prev_close: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SectorPerformance:
    """
    Sector ETF performance data.

    Attributes:
        sector: Sector name
        etf_symbol: ETF ticker symbol
        price: Current price
        change_percent: Percentage change
        volume: Trading volume
        rank: Performance rank (1 = best)
    """
    sector: str
    etf_symbol: str
    price: float
    change_percent: float
    volume: float
    rank: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MarketContext:
    """
    Complete market context snapshot.

    Attributes:
        timestamp: Snapshot timestamp
        spy: S&P 500 ETF indicator
        qqq: NASDAQ 100 ETF indicator
        vix: Volatility index
        sectors: List of sector performances
        market_regime: Market regime classification
        risk_level: Risk level (0-100)
    """
    timestamp: datetime
    spy: MarketIndicator
    qqq: MarketIndicator
    vix: MarketIndicator
    sectors: List[SectorPerformance]
    market_regime: Optional[str] = None
    risk_level: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['sectors'] = [s.to_dict() for s in self.sectors]
        return data


class MarketContextCollector:
    """
    Collector for market-wide context data.

    Provides methods to fetch and analyze market indicators including:
    - SPY (S&P 500 ETF)
    - QQQ (NASDAQ 100 ETF)
    - VIX (Volatility Index)
    - 11 Sector ETFs
    """

    # Major market indices/ETFs
    MARKET_INDICES = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'VIX': 'CBOE Volatility Index'
    }

    # Sector ETF mapping
    SECTOR_MAPPING = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services'
    }

    def __init__(self, client: Optional[PolygonClientWrapper] = None):
        """
        Initialize market context collector.

        Args:
            client: Polygon client wrapper (uses global if not provided)
        """
        self.client = client or get_polygon_client()
        logger.info("MarketContextCollector initialized")

    def get_market_indicator(self, symbol: str) -> Optional[MarketIndicator]:
        """
        Get current data for a market indicator.

        Args:
            symbol: Ticker symbol (SPY, QQQ, VIX, etc.)

        Returns:
            MarketIndicator object or None
        """
        try:
            logger.debug(f"Fetching market indicator for {symbol}")

            snapshot = self.client.get_snapshot_ticker("stocks", symbol)

            if not snapshot:
                logger.debug(f"No snapshot available for {symbol}")
                return None

            # Extract data
            day_data = snapshot.day if hasattr(snapshot, 'day') else None
            prev_day = snapshot.prevDay if hasattr(snapshot, 'prevDay') else None

            if not day_data:
                return None

            # Get current price
            price = getattr(day_data, 'c', None) or getattr(day_data, 'close', None)
            if not price and hasattr(snapshot, 'last_trade'):
                price = getattr(snapshot.last_trade, 'p', None)

            # Get volume
            volume = getattr(day_data, 'v', None) or getattr(day_data, 'volume', None) or 0

            # Get previous close
            prev_close = getattr(prev_day, 'c', None) if prev_day else None
            if not prev_close and hasattr(day_data, 'o'):
                prev_close = day_data.o

            # Calculate change percent
            if price and prev_close and prev_close > 0:
                change_percent = ((price - prev_close) / prev_close) * 100
            else:
                change_percent = getattr(snapshot, 'todaysChangePerc', 0) or 0

            if not price or not prev_close:
                return None

            indicator = MarketIndicator(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(price),
                change_percent=float(change_percent),
                volume=float(volume),
                prev_close=float(prev_close)
            )

            logger.debug(f"{symbol}: ${price:.2f} ({change_percent:+.2f}%)")

            return indicator

        except Exception as e:
            logger.error(f"Failed to get market indicator for {symbol}: {str(e)}")
            return None

    def get_sector_performance(self, etf_symbol: str) -> Optional[SectorPerformance]:
        """
        Get performance data for a sector ETF.

        Args:
            etf_symbol: Sector ETF ticker symbol

        Returns:
            SectorPerformance object or None
        """
        try:
            sector_name = self.SECTOR_MAPPING.get(etf_symbol, etf_symbol)

            indicator = self.get_market_indicator(etf_symbol)

            if not indicator:
                return None

            performance = SectorPerformance(
                sector=sector_name,
                etf_symbol=etf_symbol,
                price=indicator.price,
                change_percent=indicator.change_percent,
                volume=indicator.volume
            )

            return performance

        except Exception as e:
            logger.error(f"Failed to get sector performance for {etf_symbol}: {str(e)}")
            return None

    def get_all_sectors(self) -> List[SectorPerformance]:
        """
        Get performance data for all sector ETFs.

        Returns:
            List of SectorPerformance objects sorted by performance
        """
        sectors = []

        logger.info("Fetching all sector ETF data...")

        for etf_symbol in self.SECTOR_MAPPING.keys():
            performance = self.get_sector_performance(etf_symbol)
            if performance:
                sectors.append(performance)

        # Sort by performance (descending)
        sectors.sort(key=lambda x: x.change_percent, reverse=True)

        # Add rankings
        for rank, sector in enumerate(sectors, start=1):
            sector.rank = rank

        logger.info(
            f"Fetched {len(sectors)}/11 sector ETFs. "
            f"Top: {sectors[0].sector if sectors else 'N/A'}"
        )

        return sectors

    def get_market_context(self) -> Optional[MarketContext]:
        """
        Get complete market context snapshot.

        Returns:
            MarketContext object with all market indicators
        """
        try:
            logger.info("Collecting market context data...")

            # Get major indices
            spy = self.get_market_indicator('SPY')
            qqq = self.get_market_indicator('QQQ')
            vix = self.get_market_indicator('VIX')

            # Check if we have essential data
            if not spy or not qqq:
                logger.warning("Missing essential market data (SPY/QQQ)")
                return None

            # Get sector data
            sectors = self.get_all_sectors()

            # Determine market regime
            market_regime = self._classify_market_regime(spy, qqq, vix)

            # Calculate risk level
            risk_level = self._calculate_risk_level(spy, qqq, vix, sectors)

            context = MarketContext(
                timestamp=datetime.now(),
                spy=spy,
                qqq=qqq,
                vix=vix or self._create_fallback_vix(),
                sectors=sectors,
                market_regime=market_regime,
                risk_level=risk_level
            )

            logger.info(
                f"Market context: SPY {spy.change_percent:+.2f}%, "
                f"QQQ {qqq.change_percent:+.2f}%, "
                f"VIX ${vix.price:.2f if vix else 'N/A'}, "
                f"Regime: {market_regime}"
            )

            return context

        except Exception as e:
            logger.error(f"Failed to get market context: {str(e)}")
            return None

    def _classify_market_regime(
        self,
        spy: MarketIndicator,
        qqq: MarketIndicator,
        vix: Optional[MarketIndicator]
    ) -> str:
        """
        Classify current market regime.

        Args:
            spy: SPY indicator
            qqq: QQQ indicator
            vix: VIX indicator (optional)

        Returns:
            Market regime classification
        """
        try:
            spy_change = spy.change_percent
            qqq_change = qqq.change_percent
            vix_level = vix.price if vix else None

            # Strong uptrend
            if spy_change > 1.0 and qqq_change > 1.0:
                return "strong_uptrend"

            # Moderate uptrend
            if spy_change > 0.3 and qqq_change > 0.3:
                return "uptrend"

            # Strong downtrend
            if spy_change < -1.0 and qqq_change < -1.0:
                return "strong_downtrend"

            # Moderate downtrend
            if spy_change < -0.3 and qqq_change < -0.3:
                return "downtrend"

            # High volatility (if VIX available)
            if vix_level and vix_level > 30:
                return "high_volatility"

            # Divergence (SPY up, QQQ down or vice versa)
            if (spy_change > 0.5 and qqq_change < -0.5) or (spy_change < -0.5 and qqq_change > 0.5):
                return "divergence"

            # Default: ranging
            return "ranging"

        except Exception as e:
            logger.debug(f"Failed to classify market regime: {str(e)}")
            return "unknown"

    def _calculate_risk_level(
        self,
        spy: MarketIndicator,
        qqq: MarketIndicator,
        vix: Optional[MarketIndicator],
        sectors: List[SectorPerformance]
    ) -> float:
        """
        Calculate market risk level (0-100).

        Higher values indicate higher risk.

        Args:
            spy: SPY indicator
            qqq: QQQ indicator
            vix: VIX indicator
            sectors: List of sector performances

        Returns:
            Risk level score (0-100)
        """
        try:
            risk_components = []

            # 1. VIX level (if available)
            if vix:
                # Normalize VIX (typical range: 10-80)
                vix_risk = min(100, (vix.price / 50) * 100)
                risk_components.append(vix_risk * 0.4)  # 40% weight

            # 2. Market decline
            avg_decline = (abs(min(0, spy.change_percent)) + abs(min(0, qqq.change_percent))) / 2
            decline_risk = min(100, avg_decline * 20)  # 5% decline = 100 risk
            risk_components.append(decline_risk * 0.3)  # 30% weight

            # 3. Sector dispersion
            if sectors and len(sectors) >= 2:
                sector_changes = [s.change_percent for s in sectors]
                sector_std = pd.Series(sector_changes).std()
                dispersion_risk = min(100, sector_std * 20)
                risk_components.append(dispersion_risk * 0.2)  # 20% weight

            # 4. Divergence between SPY and QQQ
            divergence = abs(spy.change_percent - qqq.change_percent)
            divergence_risk = min(100, divergence * 30)
            risk_components.append(divergence_risk * 0.1)  # 10% weight

            # Calculate weighted average
            if risk_components:
                total_risk = sum(risk_components)
                return round(total_risk, 2)
            else:
                return 50.0  # Neutral if no data

        except Exception as e:
            logger.debug(f"Failed to calculate risk level: {str(e)}")
            return 50.0

    def _create_fallback_vix(self) -> MarketIndicator:
        """
        Create a fallback VIX indicator when data is unavailable.

        Returns:
            MarketIndicator with neutral VIX values
        """
        return MarketIndicator(
            symbol='VIX',
            timestamp=datetime.now(),
            price=20.0,  # Neutral VIX level
            change_percent=0.0,
            volume=0.0,
            prev_close=20.0
        )

    def get_market_breadth(self) -> Dict[str, float]:
        """
        Calculate market breadth indicators.

        Returns:
            Dictionary with breadth metrics
        """
        sectors = self.get_all_sectors()

        if not sectors:
            return {}

        # Count advancing/declining sectors
        advancing = sum(1 for s in sectors if s.change_percent > 0)
        declining = sum(1 for s in sectors if s.change_percent < 0)
        unchanged = len(sectors) - advancing - declining

        # Calculate breadth ratio
        total = advancing + declining
        breadth_ratio = (advancing / total) if total > 0 else 0.5

        # Average sector performance
        avg_performance = sum(s.change_percent for s in sectors) / len(sectors)

        # Sector rotation (std dev of sector performance)
        sector_changes = [s.change_percent for s in sectors]
        rotation_intensity = pd.Series(sector_changes).std()

        return {
            'advancing_sectors': advancing,
            'declining_sectors': declining,
            'unchanged_sectors': unchanged,
            'breadth_ratio': breadth_ratio,
            'avg_sector_performance': avg_performance,
            'rotation_intensity': rotation_intensity,
            'total_sectors': len(sectors)
        }

    def get_leading_lagging_sectors(
        self,
        top_n: int = 3
    ) -> Dict[str, List[str]]:
        """
        Get leading and lagging sectors.

        Args:
            top_n: Number of top/bottom sectors to return

        Returns:
            Dictionary with 'leading' and 'lagging' sector lists
        """
        sectors = self.get_all_sectors()

        if not sectors:
            return {'leading': [], 'lagging': []}

        # Already sorted by performance (descending)
        leading = [s.sector for s in sectors[:top_n]]
        lagging = [s.sector for s in sectors[-top_n:]]

        return {
            'leading': leading,
            'lagging': lagging
        }

    def is_market_open(self) -> bool:
        """
        Check if market is currently open (basic check).

        Returns:
            True if likely open, False otherwise
        """
        now = datetime.now()

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check market hours (9:30 AM - 4:00 PM ET)
        # Note: This is a simplified check without timezone handling
        market_open = now.replace(
            hour=settings.MARKET_OPEN_HOUR,
            minute=settings.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = now.replace(
            hour=settings.MARKET_CLOSE_HOUR,
            minute=settings.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )

        return market_open <= now <= market_close

    def get_context_features(self) -> Optional[Dict[str, float]]:
        """
        Get market context as feature dictionary for ML models.

        Returns:
            Dictionary of market context features
        """
        context = self.get_market_context()

        if not context:
            return None

        # Build feature dictionary
        features = {
            'spy_return': context.spy.change_percent,
            'qqq_return': context.qqq.change_percent,
            'vix_level': context.vix.price,
            'vix_change': context.vix.change_percent,
            'risk_level': context.risk_level or 50.0,
        }

        # Add sector features
        if context.sectors:
            # Top 3 sector performances
            for i, sector in enumerate(context.sectors[:3], 1):
                features[f'top_sector_{i}_return'] = sector.change_percent

            # Average sector performance
            features['avg_sector_return'] = sum(s.change_percent for s in context.sectors) / len(context.sectors)

        # Add breadth features
        breadth = self.get_market_breadth()
        if breadth:
            features['breadth_ratio'] = breadth.get('breadth_ratio', 0.5)
            features['rotation_intensity'] = breadth.get('rotation_intensity', 0.0)

        # Market regime encoding (one-hot style)
        regime_map = {
            'strong_uptrend': 2,
            'uptrend': 1,
            'ranging': 0,
            'downtrend': -1,
            'strong_downtrend': -2,
            'high_volatility': -1,
            'divergence': 0
        }
        features['market_regime_score'] = regime_map.get(context.market_regime, 0)

        return features
