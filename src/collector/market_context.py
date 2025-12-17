"""Market context data collection module using Finnhub API."""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from loguru import logger

from src.collector.finnhub_client import get_finnhub_client, FinnhubClientWrapper

# Try to import yfinance for volume data (Finnhub doesn't provide volume in quotes)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Volume data will not be available.")


class SectorETF(Enum):
    """Major sector ETFs for market context (limited to 6 for API quota)."""
    XLK = "XLK"  # Technology
    XLF = "XLF"  # Financials
    XLV = "XLV"  # Healthcare
    XLE = "XLE"  # Energy
    XLI = "XLI"  # Industrials
    XLP = "XLP"  # Consumer Staples


@dataclass
class MarketIndicator:
    """
    Market-wide indicator data.

    Attributes:
        symbol: Ticker symbol (SPY, QQQ, VXX, etc.)
        timestamp: Data timestamp
        price: Current price
        change_percent: Percentage change from previous close
        volume: Trading volume (not available in Finnhub quote)
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
        vxx: VIX ETF proxy (VXX)
        sectors: List of sector performances (6 major sectors)
        market_regime: Market regime classification
        risk_level: Risk level (0-100)
    """
    timestamp: datetime
    spy: MarketIndicator
    qqq: MarketIndicator
    vxx: MarketIndicator
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
    Collector for market-wide context data using Finnhub API.

    Provides methods to fetch and analyze market indicators including:
    - SPY (S&P 500 ETF)
    - QQQ (NASDAQ 100 ETF)
    - VXX (VIX ETF proxy - since VIX direct data may be limited)
    - 6 Major Sector ETFs (limited for API quota management)
    """

    # Major market indices/ETFs
    MARKET_INDICES = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'VXX': 'VIX ETF (Volatility Proxy)'
    }

    # Sector ETF mapping (limited to 6 for API quota)
    SECTOR_MAPPING = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples'
    }

    def __init__(self, client: Optional[FinnhubClientWrapper] = None):
        """
        Initialize market context collector.

        Args:
            client: Finnhub client wrapper (uses global if not provided)
        """
        self.client = client or get_finnhub_client()
        self._volume_cache: Dict[str, tuple] = {}  # Cache: symbol -> (volume, timestamp)
        self._volume_cache_ttl = 60  # Cache TTL in seconds
        logger.info("MarketContextCollector initialized with Finnhub")

    def _get_volume_from_yfinance(self, symbol: str) -> float:
        """
        Get current day's trading volume from yfinance.

        Args:
            symbol: Ticker symbol

        Returns:
            Trading volume or 0.0 if unavailable
        """
        if not HAS_YFINANCE:
            return 0.0

        try:
            # Check cache first
            now = datetime.now()
            if symbol in self._volume_cache:
                cached_volume, cached_time = self._volume_cache[symbol]
                if (now - cached_time).seconds < self._volume_cache_ttl:
                    return cached_volume

            # Fetch from yfinance (get last 1 day of data)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")

            if hist.empty:
                logger.debug(f"No volume data from yfinance for {symbol}")
                return 0.0

            # Get the latest volume
            volume = float(hist['Volume'].iloc[-1])

            # Cache the result
            self._volume_cache[symbol] = (volume, now)

            logger.debug(f"Got volume from yfinance for {symbol}: {volume:,.0f}")
            return volume

        except Exception as e:
            logger.debug(f"Failed to get volume from yfinance for {symbol}: {e}")
            return 0.0

    def get_market_indicator(self, symbol: str) -> Optional[MarketIndicator]:
        """
        Get current data for a market indicator.

        Args:
            symbol: Ticker symbol (SPY, QQQ, VXX, etc.)

        Returns:
            MarketIndicator object or None
        """
        try:
            logger.debug(f"Fetching market indicator for {symbol}")

            # Get quote from Finnhub
            quote = self.client.get_quote(symbol)

            if not quote:
                logger.debug(f"No quote available for {symbol}")
                return None

            # Extract data from Finnhub quote
            # Keys: c (current), h (high), l (low), o (open), pc (prev close), t (timestamp)
            current_price = quote.get('c')
            prev_close = quote.get('pc')
            timestamp = quote.get('t')

            if current_price is None or prev_close is None:
                logger.debug(f"Incomplete data for {symbol}")
                return None

            # Calculate change percent
            if prev_close > 0:
                change_percent = ((current_price - prev_close) / prev_close) * 100
            else:
                change_percent = 0.0

            # Get volume from yfinance (Finnhub quote doesn't provide volume)
            volume = self._get_volume_from_yfinance(symbol)

            indicator = MarketIndicator(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(timestamp) if timestamp else datetime.now(),
                price=float(current_price),
                change_percent=float(change_percent),
                volume=volume,
                prev_close=float(prev_close)
            )

            logger.debug(f"{symbol}: ${current_price:.2f} ({change_percent:+.2f}%)")

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
        Get performance data for all tracked sector ETFs (6 major sectors).

        Returns:
            List of SectorPerformance objects sorted by performance
        """
        sectors = []

        logger.info("Fetching sector ETF data (6 sectors)...")

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
            f"Fetched {len(sectors)}/6 sector ETFs. "
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
            vxx = self.get_market_indicator('VXX')  # VIX proxy

            # Check if we have essential data
            if not spy or not qqq:
                logger.warning("Missing essential market data (SPY/QQQ)")
                return None

            # Get sector data
            sectors = self.get_all_sectors()

            # Determine market regime
            market_regime = self._classify_market_regime(spy, qqq, vxx)

            # Calculate risk level
            risk_level = self._calculate_risk_level(spy, qqq, vxx, sectors)

            context = MarketContext(
                timestamp=datetime.now(),
                spy=spy,
                qqq=qqq,
                vxx=vxx or self._create_fallback_vxx(),
                sectors=sectors,
                market_regime=market_regime,
                risk_level=risk_level
            )

            logger.info(
                f"Market context: SPY {spy.change_percent:+.2f}%, "
                f"QQQ {qqq.change_percent:+.2f}%, "
                f"VXX ${vxx.price:.2f if vxx else 'N/A'}, "
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
        vxx: Optional[MarketIndicator]
    ) -> str:
        """
        Classify current market regime (simplified).

        Args:
            spy: SPY indicator
            qqq: QQQ indicator
            vxx: VXX indicator (VIX proxy)

        Returns:
            Market regime classification
        """
        try:
            spy_change = spy.change_percent
            qqq_change = qqq.change_percent

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

            # High volatility (VXX rising significantly)
            if vxx and vxx.change_percent > 10:
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
        vxx: Optional[MarketIndicator],
        sectors: List[SectorPerformance]
    ) -> float:
        """
        Calculate market risk level (0-100) - simplified version.

        Higher values indicate higher risk.

        Args:
            spy: SPY indicator
            qqq: QQQ indicator
            vxx: VXX indicator (VIX proxy)
            sectors: List of sector performances

        Returns:
            Risk level score (0-100)
        """
        try:
            risk_components = []

            # 1. VXX level (if available) - proxy for VIX
            if vxx:
                # VXX typical range: $10-50, normalize to 0-100
                vxx_risk = min(100, (vxx.price / 40) * 100)
                risk_components.append(vxx_risk * 0.4)  # 40% weight

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

    def _create_fallback_vxx(self) -> MarketIndicator:
        """
        Create a fallback VXX indicator when data is unavailable.

        Returns:
            MarketIndicator with neutral VXX values
        """
        return MarketIndicator(
            symbol='VXX',
            timestamp=datetime.now(),
            price=20.0,  # Neutral VXX level
            change_percent=0.0,
            volume=0.0,
            prev_close=20.0
        )

    def get_market_breadth(self) -> Dict[str, float]:
        """
        Calculate market breadth indicators from sector data.

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
        leading = [s.sector for s in sectors[:min(top_n, len(sectors))]]
        lagging = [s.sector for s in sectors[-min(top_n, len(sectors)):]]

        return {
            'leading': leading,
            'lagging': lagging
        }

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
            'vxx_level': context.vxx.price,
            'vxx_change': context.vxx.change_percent,
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

        # Market regime encoding
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
