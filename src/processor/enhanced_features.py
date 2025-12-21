"""
Enhanced Feature Engineering Module

Adds new feature categories:
1. Sector Relative Strength - stock performance vs sector ETF
2. Market Regime - market state classification
3. Calendar Features - earnings, FOMC, month-end, etc.
4. Cross-Market Features - bonds, dollar, gold signals
"""

from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from loguru import logger

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

warnings.filterwarnings('ignore')


# FOMC Meeting Dates 2024-2025 (announcement dates)
FOMC_DATES_2024 = [
    datetime(2024, 1, 31), datetime(2024, 3, 20), datetime(2024, 5, 1),
    datetime(2024, 6, 12), datetime(2024, 7, 31), datetime(2024, 9, 18),
    datetime(2024, 11, 7), datetime(2024, 12, 18),
]

FOMC_DATES_2025 = [
    datetime(2025, 1, 29), datetime(2025, 3, 19), datetime(2025, 5, 7),
    datetime(2025, 6, 18), datetime(2025, 7, 30), datetime(2025, 9, 17),
    datetime(2025, 11, 5), datetime(2025, 12, 17),
]

FOMC_DATES = set(d.date() for d in FOMC_DATES_2024 + FOMC_DATES_2025)


# Sector ETF to GICS Sector mapping
SECTOR_ETF_MAP = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}

# Stock to Sector mapping (major NASDAQ stocks)
STOCK_SECTOR_MAP = {
    # Technology
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AVGO': 'XLK', 'ASML': 'XLK',
    'AMD': 'XLK', 'INTC': 'XLK', 'QCOM': 'XLK', 'TXN': 'XLK', 'MU': 'XLK',
    'AMAT': 'XLK', 'LRCX': 'XLK', 'KLAC': 'XLK', 'MCHP': 'XLK', 'SNPS': 'XLK',
    'MRVL': 'XLK', 'NXPI': 'XLK', 'ON': 'XLK', 'FTNT': 'XLK', 'CRWD': 'XLK',
    'PANW': 'XLK', 'ADBE': 'XLK', 'CRM': 'XLK', 'NOW': 'XLK', 'PLTR': 'XLK',

    # Communication Services
    'GOOGL': 'XLC', 'GOOG': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'TMUS': 'XLC',
    'CMCSA': 'XLC', 'EA': 'XLC', 'TTWO': 'XLC',

    # Consumer Discretionary
    'AMZN': 'XLY', 'TSLA': 'XLY', 'SBUX': 'XLY', 'LULU': 'XLY', 'ORLY': 'XLY',
    'ROST': 'XLY', 'EBAY': 'XLY', 'MAR': 'XLY', 'ABNB': 'XLY', 'BKNG': 'XLY',
    'CPRT': 'XLY', 'FAST': 'XLY', 'PCAR': 'XLY',

    # Healthcare
    'AMGN': 'XLV', 'GILD': 'XLV', 'VRTX': 'XLV', 'REGN': 'XLV', 'BIIB': 'XLV',
    'ILMN': 'XLV', 'ISRG': 'XLV', 'MRNA': 'XLV', 'DXCM': 'XLV', 'IDXX': 'XLV',

    # Consumer Staples
    'PEP': 'XLP', 'COST': 'XLP', 'MNST': 'XLP', 'KDP': 'XLP', 'WBA': 'XLP',

    # Financials
    'PYPL': 'XLF', 'COIN': 'XLF', 'HOOD': 'XLF',

    # Industrials
    'HON': 'XLI', 'CSX': 'XLI', 'ODFL': 'XLI', 'PAYX': 'XLI',

    # Energy
    'ENPH': 'XLE', 'FSLR': 'XLE',

    # Utilities
    'XEL': 'XLU', 'EXC': 'XLU',
}


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering with sector, market regime, calendar, and cross-market features.
    """

    # Reference ETFs for cross-market analysis
    REFERENCE_ETFS = ['SPY', 'QQQ', 'TLT', 'GLD', 'UUP', 'VXX']
    SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP']

    def __init__(self, cache_days: int = 90):
        """
        Initialize enhanced feature engineer.

        Args:
            cache_days: Number of days to cache reference ETF data
        """
        self.cache_days = cache_days
        self._etf_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Optional[datetime] = None

    def _load_etf_data(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """Load ETF data from Yahoo Finance."""
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, skipping ETF data")
            return {}

        result = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='5m')

                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    if 'datetime' in df.columns:
                        df = df.rename(columns={'datetime': 'timestamp'})
                    result[symbol] = df
                    logger.debug(f"Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.debug(f"Failed to load {symbol}: {e}")

        return result

    def _ensure_cache(self, days: int = 60):
        """Ensure ETF cache is loaded and fresh."""
        now = datetime.now()

        # Check if cache is stale (older than 1 hour)
        if self._cache_timestamp and (now - self._cache_timestamp).seconds < 3600:
            return

        logger.info("Loading reference ETF data...")

        # Load reference ETFs
        all_symbols = self.REFERENCE_ETFS + self.SECTOR_ETFS
        self._etf_cache = self._load_etf_data(all_symbols, days)
        self._cache_timestamp = now

        logger.info(f"Loaded {len(self._etf_cache)} ETFs into cache")

    def _align_etf_data(self, stock_df: pd.DataFrame, etf_symbol: str) -> Optional[pd.Series]:
        """Align ETF data to stock timestamps."""
        if etf_symbol not in self._etf_cache:
            return None

        etf_df = self._etf_cache[etf_symbol].copy()

        if 'timestamp' not in etf_df.columns or 'timestamp' not in stock_df.columns:
            return None

        # Ensure timezone-naive
        etf_df['timestamp'] = pd.to_datetime(etf_df['timestamp']).dt.tz_localize(None)
        stock_timestamps = pd.to_datetime(stock_df['timestamp']).dt.tz_localize(None)

        # Create a merged dataframe
        etf_df = etf_df.set_index('timestamp')

        # Reindex to stock timestamps using forward fill
        aligned = etf_df['close'].reindex(stock_timestamps, method='ffill')

        return aligned.values

    def compute_sector_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Compute sector relative strength features.

        Features:
        - sector_return_5m: Sector ETF 5-bar return
        - sector_return_15m: Sector ETF 15-bar return
        - stock_vs_sector: Stock return minus sector return
        - sector_rank: Sector performance rank (1-6)
        """
        df = df.copy()

        # Get sector ETF for this stock
        sector_etf = STOCK_SECTOR_MAP.get(ticker, 'XLK')  # Default to tech

        # Initialize with defaults
        df['sector_return_5m'] = 0.0
        df['sector_return_15m'] = 0.0
        df['stock_vs_sector'] = 0.0
        df['sector_rank'] = 3.0  # Middle rank

        # Get sector ETF data
        sector_prices = self._align_etf_data(df, sector_etf)

        if sector_prices is not None and len(sector_prices) == len(df):
            sector_series = pd.Series(sector_prices)

            # Sector returns
            df['sector_return_5m'] = sector_series.pct_change(5).fillna(0)
            df['sector_return_15m'] = sector_series.pct_change(15).fillna(0)

            # Stock vs sector (alpha)
            stock_return = df['close'].pct_change(5).fillna(0)
            df['stock_vs_sector'] = stock_return - df['sector_return_5m']

        # Calculate sector rank based on all sector ETFs
        sector_returns = {}
        for etf in self.SECTOR_ETFS:
            etf_prices = self._align_etf_data(df, etf)
            if etf_prices is not None and len(etf_prices) > 5:
                returns = pd.Series(etf_prices).pct_change(5).fillna(0)
                sector_returns[etf] = returns

        if sector_returns and sector_etf in sector_returns:
            # Compute rank for each timestamp
            returns_df = pd.DataFrame(sector_returns)
            ranks = returns_df.rank(axis=1, ascending=False)
            if sector_etf in ranks.columns:
                df['sector_rank'] = ranks[sector_etf].values

        return df

    def compute_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market regime features.

        Features:
        - spy_return_5m: SPY 5-bar return
        - qqq_return_5m: QQQ 5-bar return
        - market_regime: Regime score (-2 to +2)
        - spy_qqq_divergence: Divergence between SPY and QQQ
        - vix_level: VXX price (proxy for VIX)
        - vix_change: VXX change
        """
        df = df.copy()

        # Initialize with defaults
        df['spy_return_5m'] = 0.0
        df['qqq_return_5m'] = 0.0
        df['market_regime'] = 0.0
        df['spy_qqq_divergence'] = 0.0
        df['vix_proxy'] = 20.0
        df['vix_change'] = 0.0

        # SPY features
        spy_prices = self._align_etf_data(df, 'SPY')
        if spy_prices is not None and len(spy_prices) == len(df):
            spy_series = pd.Series(spy_prices)
            df['spy_return_5m'] = spy_series.pct_change(5).fillna(0) * 100  # Percent

        # QQQ features
        qqq_prices = self._align_etf_data(df, 'QQQ')
        if qqq_prices is not None and len(qqq_prices) == len(df):
            qqq_series = pd.Series(qqq_prices)
            df['qqq_return_5m'] = qqq_series.pct_change(5).fillna(0) * 100  # Percent

        # Divergence
        df['spy_qqq_divergence'] = df['spy_return_5m'] - df['qqq_return_5m']

        # Market regime classification
        spy_ret = df['spy_return_5m']
        qqq_ret = df['qqq_return_5m']

        conditions = [
            (spy_ret > 0.5) & (qqq_ret > 0.5),   # Strong uptrend
            (spy_ret > 0.1) & (qqq_ret > 0.1),   # Uptrend
            (spy_ret < -0.5) & (qqq_ret < -0.5), # Strong downtrend
            (spy_ret < -0.1) & (qqq_ret < -0.1), # Downtrend
        ]
        choices = [2, 1, -2, -1]
        df['market_regime'] = np.select(conditions, choices, default=0)

        # VXX (VIX proxy)
        vxx_prices = self._align_etf_data(df, 'VXX')
        if vxx_prices is not None and len(vxx_prices) == len(df):
            vxx_series = pd.Series(vxx_prices)
            df['vix_proxy'] = vxx_series.fillna(20.0)
            df['vix_change'] = vxx_series.pct_change(5).fillna(0) * 100

        return df

    def compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute calendar-based features.

        Features:
        - is_fomc_day: FOMC announcement day
        - is_fomc_week: Week of FOMC meeting
        - is_month_end: Last 3 trading days of month
        - is_month_start: First 3 trading days of month
        - is_quarter_end: Last week of quarter
        - is_friday: Friday (options expiration effects)
        - hour_of_day: Hour (0-23)
        - is_power_hour: Last hour of trading (3-4 PM)
        - is_opening_hour: First hour of trading (9:30-10:30)
        """
        df = df.copy()

        if 'timestamp' not in df.columns:
            # Set defaults
            for col in ['is_fomc_day', 'is_fomc_week', 'is_month_end', 'is_month_start',
                       'is_quarter_end', 'is_friday', 'hour_of_day', 'is_power_hour', 'is_opening_hour']:
                df[col] = 0
            return df

        timestamps = pd.to_datetime(df['timestamp'])
        dates = timestamps.dt.date

        # FOMC day
        df['is_fomc_day'] = dates.apply(lambda d: 1 if d in FOMC_DATES else 0)

        # FOMC week (3 days before and after)
        def is_fomc_week(d):
            for fomc_date in FOMC_DATES:
                if abs((d - fomc_date).days) <= 3:
                    return 1
            return 0
        df['is_fomc_week'] = dates.apply(is_fomc_week)

        # Month end (last 3 days)
        df['is_month_end'] = ((timestamps.dt.day >= 28) |
                              ((timestamps + pd.Timedelta(days=3)).dt.month != timestamps.dt.month)).astype(int)

        # Month start (first 3 days)
        df['is_month_start'] = (timestamps.dt.day <= 3).astype(int)

        # Quarter end
        df['is_quarter_end'] = ((timestamps.dt.month.isin([3, 6, 9, 12])) &
                                (timestamps.dt.day >= 25)).astype(int)

        # Friday
        df['is_friday'] = (timestamps.dt.dayofweek == 4).astype(int)

        # Hour of day
        df['hour_of_day'] = timestamps.dt.hour

        # Power hour (3-4 PM)
        df['is_power_hour'] = ((timestamps.dt.hour == 15) |
                               ((timestamps.dt.hour == 15) & (timestamps.dt.minute >= 30))).astype(int)

        # Opening hour (9:30-10:30)
        df['is_opening_hour'] = (((timestamps.dt.hour == 9) & (timestamps.dt.minute >= 30)) |
                                  (timestamps.dt.hour == 10)).astype(int)

        return df

    def compute_cross_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-market features.

        Features:
        - bond_return: TLT (20+ year treasury) return
        - gold_return: GLD return
        - dollar_return: UUP (dollar index) return
        - stock_bond_corr: Rolling correlation with bonds
        - risk_on_signal: Risk-on/off signal
        """
        df = df.copy()

        # Initialize defaults
        df['bond_return'] = 0.0
        df['gold_return'] = 0.0
        df['dollar_return'] = 0.0
        df['risk_on_signal'] = 0.0

        # TLT (Bonds)
        tlt_prices = self._align_etf_data(df, 'TLT')
        if tlt_prices is not None and len(tlt_prices) == len(df):
            tlt_series = pd.Series(tlt_prices)
            df['bond_return'] = tlt_series.pct_change(5).fillna(0) * 100

        # GLD (Gold)
        gld_prices = self._align_etf_data(df, 'GLD')
        if gld_prices is not None and len(gld_prices) == len(df):
            gld_series = pd.Series(gld_prices)
            df['gold_return'] = gld_series.pct_change(5).fillna(0) * 100

        # UUP (Dollar)
        uup_prices = self._align_etf_data(df, 'UUP')
        if uup_prices is not None and len(uup_prices) == len(df):
            uup_series = pd.Series(uup_prices)
            df['dollar_return'] = uup_series.pct_change(5).fillna(0) * 100

        # Risk-on signal: SPY up + TLT down = risk-on, opposite = risk-off
        spy_ret = df.get('spy_return_5m', pd.Series([0] * len(df)))
        bond_ret = df['bond_return']

        # Risk-on when stocks up and bonds down (or vice versa)
        df['risk_on_signal'] = np.where(
            (spy_ret > 0) & (bond_ret < 0), 1,  # Risk-on
            np.where((spy_ret < 0) & (bond_ret > 0), -1, 0)  # Risk-off
        )

        return df

    def compute_all_enhanced_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        include_sector: bool = True,
        include_regime: bool = True,
        include_calendar: bool = True,
        include_cross_market: bool = True
    ) -> pd.DataFrame:
        """
        Compute all enhanced features.

        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            include_sector: Include sector features
            include_regime: Include market regime features
            include_calendar: Include calendar features
            include_cross_market: Include cross-market features

        Returns:
            DataFrame with all enhanced features added
        """
        # Ensure cache is loaded
        self._ensure_cache()

        df = df.copy()

        if include_regime:
            df = self.compute_market_regime_features(df)

        if include_sector:
            df = self.compute_sector_features(df, ticker)

        if include_calendar:
            df = self.compute_calendar_features(df)

        if include_cross_market:
            df = self.compute_cross_market_features(df)

        return df

    def get_enhanced_feature_names(self) -> List[str]:
        """Get list of all enhanced feature names."""
        return [
            # Sector features (4)
            'sector_return_5m', 'sector_return_15m', 'stock_vs_sector', 'sector_rank',

            # Market regime features (6)
            'spy_return_5m', 'qqq_return_5m', 'market_regime',
            'spy_qqq_divergence', 'vix_proxy', 'vix_change',

            # Calendar features (9)
            'is_fomc_day', 'is_fomc_week', 'is_month_end', 'is_month_start',
            'is_quarter_end', 'is_friday', 'hour_of_day', 'is_power_hour', 'is_opening_hour',

            # Cross-market features (4)
            'bond_return', 'gold_return', 'dollar_return', 'risk_on_signal',
        ]


# Singleton instance
_enhanced_engineer = None


def get_enhanced_feature_engineer() -> EnhancedFeatureEngineer:
    """Get global enhanced feature engineer instance."""
    global _enhanced_engineer
    if _enhanced_engineer is None:
        _enhanced_engineer = EnhancedFeatureEngineer()
    return _enhanced_engineer
