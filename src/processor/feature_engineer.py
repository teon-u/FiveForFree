"""
Feature Engineering Module for NASDAQ Prediction System

Generates 49 features across 7 categories:
- Price-based (15)
- Volatility-based (10)
- Volume-based (8)
- Order book (0)
- Momentum (8)
- Market context (5)
- Time-based (3)
"""

from datetime import datetime, time
from typing import Dict, Optional, List, Tuple
import warnings
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd
from loguru import logger

# Try to import talib, fallback to pandas implementation if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

warnings.filterwarnings('ignore')


# Fallback implementations for TA-Lib functions using pandas
def _sma(series: np.ndarray, timeperiod: int) -> np.ndarray:
    """Simple Moving Average."""
    if HAS_TALIB:
        return talib.SMA(series, timeperiod=timeperiod)
    return pd.Series(series).rolling(window=timeperiod).mean().values


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Average True Range."""
    if HAS_TALIB:
        return talib.ATR(high, low, close, timeperiod=timeperiod)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)

    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=timeperiod).mean().values


def _bbands(close: np.ndarray, timeperiod: int, nbdevup: float, nbdevdn: float):
    """Bollinger Bands."""
    if HAS_TALIB:
        return talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)

    close_s = pd.Series(close)
    middle = close_s.rolling(window=timeperiod).mean()
    std = close_s.rolling(window=timeperiod).std()
    upper = middle + nbdevup * std
    lower = middle - nbdevdn * std

    return upper.values, middle.values, lower.values


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    if HAS_TALIB:
        return talib.OBV(close, volume)

    close_s = pd.Series(close)
    volume_s = pd.Series(volume)

    direction = np.where(close_s.diff() > 0, 1, np.where(close_s.diff() < 0, -1, 0))
    return (direction * volume_s).cumsum().values


def _mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timeperiod: int) -> np.ndarray:
    """Money Flow Index."""
    if HAS_TALIB:
        return talib.MFI(high, low, close, volume, timeperiod=timeperiod)

    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    money_flow = typical_price * pd.Series(volume)

    tp_diff = typical_price.diff()
    positive_flow = money_flow.where(tp_diff > 0, 0).rolling(window=timeperiod).sum()
    negative_flow = money_flow.where(tp_diff < 0, 0).abs().rolling(window=timeperiod).sum()

    mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
    return mfi.values


def _rsi(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Relative Strength Index."""
    if HAS_TALIB:
        return talib.RSI(close, timeperiod=timeperiod)

    close_s = pd.Series(close)
    delta = close_s.diff()

    gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()

    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi.values


def _macd(close: np.ndarray, fastperiod: int, slowperiod: int, signalperiod: int):
    """Moving Average Convergence Divergence."""
    if HAS_TALIB:
        return talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fastperiod, adjust=False).mean()
    ema_slow = close_s.ewm(span=slowperiod, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line.values, signal_line.values, histogram.values


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           fastk_period: int, slowk_period: int, slowd_period: int):
    """Stochastic Oscillator."""
    if HAS_TALIB:
        return talib.STOCH(high, low, close, fastk_period=fastk_period,
                          slowk_period=slowk_period, slowd_period=slowd_period)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)

    lowest_low = low_s.rolling(window=fastk_period).min()
    highest_high = high_s.rolling(window=fastk_period).max()

    fastk = 100 * (close_s - lowest_low) / (highest_high - lowest_low + 1e-10)
    slowk = fastk.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()

    return slowk.values, slowd.values


def _willr(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Williams %R."""
    if HAS_TALIB:
        return talib.WILLR(high, low, close, timeperiod=timeperiod)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)

    highest_high = high_s.rolling(window=timeperiod).max()
    lowest_low = low_s.rolling(window=timeperiod).min()

    willr = -100 * (highest_high - close_s) / (highest_high - lowest_low + 1e-10)
    return willr.values


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    """Commodity Channel Index."""
    if HAS_TALIB:
        return talib.CCI(high, low, close, timeperiod=timeperiod)

    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    sma_tp = typical_price.rolling(window=timeperiod).mean()
    mean_deviation = typical_price.rolling(window=timeperiod).apply(lambda x: np.abs(x - x.mean()).mean())

    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    return cci.values


class FeatureEngineer:
    """
    Feature engineering for high-frequency NASDAQ trading.

    All features are computed using vectorized operations for performance.
    Handles missing data gracefully with forward-fill and fallback values.
    Includes caching for repeated computations on same data.
    """

    def __init__(
        self,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        cache_size: int = 100,
        cache_ttl_seconds: int = 60,
    ):
        """
        Initialize feature engineer.

        Args:
            market_open: Market opening time (default 9:30 AM)
            market_close: Market closing time (default 4:00 PM)
            cache_size: Maximum number of cached feature computations
            cache_ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.market_open = market_open
        self.market_close = market_close
        self.cache_size = cache_size
        self.cache_ttl_seconds = cache_ttl_seconds

        # Feature cache: {hash: (features_df, timestamp)}
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}

        # Option expiry dates (third Friday of each month)
        self.option_expiry_dates = self._generate_option_expiry_dates()

    def _get_data_hash(self, df: pd.DataFrame, market_data: Optional[Dict]) -> str:
        """Generate a hash key for caching based on input data."""
        # Use last timestamp and shape for quick hash
        hash_components = [
            str(df.shape),
            str(df['close'].iloc[-1]) if len(df) > 0 else "",
            str(df['timestamp'].iloc[-1]) if 'timestamp' in df.columns and len(df) > 0 else "",
        ]

        if market_data:
            hash_components.append(str(sorted(market_data.keys())))

        return hashlib.md5("".join(hash_components).encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached features if valid."""
        if cache_key not in self._cache:
            return None

        features_df, timestamp = self._cache[cache_key]
        current_time = datetime.now().timestamp()

        if current_time - timestamp > self.cache_ttl_seconds:
            # Cache expired
            del self._cache[cache_key]
            return None

        logger.debug(f"Feature cache hit for key {cache_key[:8]}...")
        return features_df.copy()

    def _add_to_cache(self, cache_key: str, features_df: pd.DataFrame) -> None:
        """Add features to cache."""
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.cache_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[cache_key] = (features_df.copy(), datetime.now().timestamp())
        logger.debug(f"Cached features for key {cache_key[:8]}...")

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")

    def _generate_option_expiry_dates(self) -> set:
        """Generate option expiry dates for current year."""
        expiry_dates = set()
        year = datetime.now().year

        for month in range(1, 13):
            # Find third Friday
            first_day = datetime(year, month, 1)
            first_friday = (4 - first_day.weekday()) % 7 + 1
            third_friday = first_friday + 14

            if third_friday <= 31:
                try:
                    expiry_date = datetime(year, month, third_friday).date()
                    expiry_dates.add(expiry_date)
                except ValueError:
                    pass

        return expiry_dates

    def compute_features(
        self,
        df: pd.DataFrame,
        order_book: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute all 49 features from minute bar data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume, vwap]
            order_book: Dict with order book data (optional)
            market_data: Dict with SPY, QQQ, VIX, sector data (optional)
            use_cache: Whether to use feature caching (default True)

        Returns:
            DataFrame with all computed features
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_data_hash(df, market_data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        df = df.copy()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # 1. Price-based features (15)
        df = self._add_price_features(df)

        # 2. Volatility-based features (10)
        df = self._add_volatility_features(df)

        # 3. Volume-based features (8)
        df = self._add_volume_features(df)

        # 4. Order book features (0)
        df = self._add_order_book_features(df, order_book)

        # 5. Momentum features (8)
        df = self._add_momentum_features(df)

        # 6. Market context features (5)
        df = self._add_market_context_features(df, market_data)

        # 7. Time-based features (3)
        df = self._add_time_features(df)

        # Fill any remaining NaN values
        df = self._handle_missing_values(df)

        # Store in cache
        if use_cache:
            self._add_to_cache(cache_key, df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 15 price-based features.

        Features:
        1-5: returns_1m, returns_5m, returns_15m, returns_30m, returns_60m
        6-8: ma_5, ma_15, ma_60
        9-11: price_vs_ma_5, price_vs_ma_15, price_vs_ma_60
        12: ma_cross_5_15
        13: price_vs_vwap
        14: price_momentum_5m
        15: price_momentum_15m
        """
        close = df['close'].values

        # 1-5: Returns at multiple timeframes
        df['returns_1m'] = df['close'].pct_change(1)
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        df['returns_30m'] = df['close'].pct_change(30)
        df['returns_60m'] = df['close'].pct_change(60)

        # 6-8: Moving averages
        df['ma_5'] = _sma(close, timeperiod=5)
        df['ma_15'] = _sma(close, timeperiod=15)
        df['ma_60'] = _sma(close, timeperiod=60)

        # 9-11: Price vs moving averages (normalized)
        df['price_vs_ma_5'] = (df['close'] - df['ma_5']) / df['ma_5']
        df['price_vs_ma_15'] = (df['close'] - df['ma_15']) / df['ma_15']
        df['price_vs_ma_60'] = (df['close'] - df['ma_60']) / df['ma_60']

        # 12: MA crossover signal
        df['ma_cross_5_15'] = (df['ma_5'] - df['ma_15']) / df['ma_15']

        # 13: Price vs VWAP
        if 'vwap' in df.columns:
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        else:
            # Calculate VWAP if not provided
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']

        # 14-15: Price momentum (rate of change of returns)
        df['price_momentum_5m'] = df['returns_5m'] - df['returns_1m']
        df['price_momentum_15m'] = df['returns_15m'] - df['returns_5m']

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 10 volatility-based features.

        Features:
        1: atr_14
        2: bb_position
        3: bb_width
        4-6: volatility_5m, volatility_15m, volatility_60m
        7: price_acceleration
        8: high_low_range
        9: intraday_range
        10: volatility_ratio
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 1: Average True Range
        df['atr_14'] = _atr(high, low, close, timeperiod=14)
        df['atr_14_normalized'] = df['atr_14'] / df['close']

        # 2-3: Bollinger Bands
        upper, middle, lower = _bbands(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # Position within Bollinger Bands (0 = lower, 0.5 = middle, 1 = upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 4-6: Rolling volatility (standard deviation of returns)
        df['volatility_5m'] = df['returns_1m'].rolling(5).std()
        df['volatility_15m'] = df['returns_1m'].rolling(15).std()
        df['volatility_60m'] = df['returns_1m'].rolling(60).std()

        # 7: Price acceleration (second derivative)
        df['price_acceleration'] = df['returns_1m'].diff()

        # 8: High-low range (normalized)
        df['high_low_range'] = (df['high'] - df['low']) / df['close']

        # 9: Intraday range from open (normalized)
        df['intraday_range'] = (df['high'] - df['low']) / df['open']

        # 10: Volatility ratio (short-term vs long-term)
        df['volatility_ratio'] = df['volatility_5m'] / (df['volatility_60m'] + 1e-10)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 8 volume-based features.

        Features:
        1: volume_ratio
        2: volume_ma_5
        3: volume_ma_15
        4: volume_trend
        5: obv (On-Balance Volume)
        6: money_flow
        7: mfi_14 (Money Flow Index)
        8: volume_price_trend
        """
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 1: Volume ratio (current vs average)
        df['volume_ma_20'] = _sma(volume, timeperiod=20)
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)

        # 2-3: Volume moving averages
        df['volume_ma_5'] = _sma(volume, timeperiod=5)
        df['volume_ma_15'] = _sma(volume, timeperiod=15)

        # 4: Volume trend
        df['volume_trend'] = (df['volume_ma_5'] - df['volume_ma_15']) / (df['volume_ma_15'] + 1e-10)

        # 5: On-Balance Volume
        df['obv'] = _obv(close, volume)
        df['obv_normalized'] = df['obv'] / df['obv'].rolling(60).mean()

        # 6: Money flow (price * volume)
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()

        # 7: Money Flow Index
        df['mfi_14'] = _mfi(high, low, close, volume, timeperiod=14)

        # 8: Volume Price Trend
        df['volume_price_trend'] = (df['close'].diff() / df['close'].shift(1)) * df['volume']
        df['vpt_cumsum'] = df['volume_price_trend'].cumsum()

        return df

    def _add_order_book_features(self, df: pd.DataFrame, order_book: Optional[Dict]) -> pd.DataFrame:
        """
        Add 8 order book features.

        Features:
        1: bid_ask_spread
        2: spread_pct
        3: imbalance
        4: bid_depth
        5: ask_depth
        6: depth_ratio
        7: depth_weighted_mid_price
        8: order_flow_imbalance
        """
        if order_book is None:
            # Set default values when order book data is not available (Finnhub doesn't provide Level 2 data)
            df['bid_ask_spread'] = 0.0
            df['spread_pct'] = 0.0
            df['imbalance'] = 0.0
            df['bid_depth'] = 0.0
            df['ask_depth'] = 0.0
            df['depth_ratio'] = 0.0
            df['depth_weighted_mid_price'] = 0.0
            df['order_flow_imbalance'] = 0.0
        else:
            # Extract order book data
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if bids and asks:
                best_bid = bids[0][0] if bids else df['close'].iloc[-1]
                best_ask = asks[0][0] if asks else df['close'].iloc[-1]

                # 1-2: Spread
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2

                df['bid_ask_spread'] = spread
                df['spread_pct'] = spread / mid_price if mid_price > 0 else 0

                # 3: Imbalance (from order book data if available)
                df['imbalance'] = order_book.get('imbalance', 0.0)

                # 4-6: Depth
                bid_total = order_book.get('bid_total_volume', sum(size for _, size in bids))
                ask_total = order_book.get('ask_total_volume', sum(size for _, size in asks))

                df['bid_depth'] = bid_total
                df['ask_depth'] = ask_total
                df['depth_ratio'] = bid_total / (ask_total + 1e-10)

                # 7: Depth-weighted mid price
                bid_value = sum(price * size for price, size in bids)
                ask_value = sum(price * size for price, size in asks)
                total_volume = bid_total + ask_total

                df['depth_weighted_mid_price'] = (bid_value + ask_value) / (total_volume + 1e-10)

                # 8: Order flow imbalance (normalized)
                df['order_flow_imbalance'] = (bid_total - ask_total) / (bid_total + ask_total + 1e-10)
            else:
                # Fallback to default values
                df['bid_ask_spread'] = 0.0
                df['spread_pct'] = 0.0
                df['imbalance'] = 0.0
                df['bid_depth'] = 0.0
                df['ask_depth'] = 0.0
                df['depth_ratio'] = 0.0
                df['depth_weighted_mid_price'] = 0.0
                df['order_flow_imbalance'] = 0.0

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 8 momentum features.

        Features:
        1: rsi_14
        2: macd
        3: macd_signal
        4: macd_hist
        5: stoch_k
        6: stoch_d
        7: williams_r
        8: cci_14
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 1: RSI (Relative Strength Index)
        df['rsi_14'] = _rsi(close, timeperiod=14)

        # 2-4: MACD
        macd, macd_signal, macd_hist = _macd(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # 5-6: Stochastic Oscillator
        stoch_k, stoch_d = _stoch(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # 7: Williams %R
        df['williams_r'] = _willr(high, low, close, timeperiod=14)

        # 8: Commodity Channel Index
        df['cci_14'] = _cci(high, low, close, timeperiod=14)

        return df

    def _add_market_context_features(self, df: pd.DataFrame, market_data: Optional[Dict]) -> pd.DataFrame:
        """
        Add 5 market context features.

        Features:
        1: spy_return
        2: qqq_return
        3: vix_level
        4: sector_etf_return
        5: market_correlation
        """
        if market_data is None:
            # Set default values when market data is not available
            df['spy_return'] = 0.0
            df['qqq_return'] = 0.0
            df['vix_level'] = 20.0  # Neutral VIX level
            df['sector_etf_return'] = 0.0
            df['market_correlation'] = 0.0
        else:
            # 1: SPY return
            df['spy_return'] = market_data.get('spy_return', 0.0)

            # 2: QQQ return
            df['qqq_return'] = market_data.get('qqq_return', 0.0)

            # 3: VIX level (normalized)
            vix = market_data.get('vix_level', 20.0)
            df['vix_level'] = vix / 100.0  # Normalize to 0-1 range

            # 4: Sector ETF return
            df['sector_etf_return'] = market_data.get('sector_return', 0.0)

            # 5: Market correlation (correlation with SPY/QQQ)
            # Calculate using historical returns if available
            market_correlation = self._calculate_market_correlation(
                df, market_data
            )
            df['market_correlation'] = market_correlation

        return df

    def _calculate_market_correlation(
        self,
        df: pd.DataFrame,
        market_data: Optional[Dict],
        window: int = 30
    ) -> float:
        """
        Calculate rolling correlation between stock returns and market (SPY/QQQ).

        Args:
            df: DataFrame with stock price data (must have 'close' column)
            market_data: Dict containing 'spy_prices' and/or 'qqq_prices' arrays
            window: Rolling window for correlation calculation (default 30 periods)

        Returns:
            Correlation coefficient between -1 and 1, or 0.0 if cannot calculate
        """
        # Check if we have enough data
        if df is None or len(df) < window:
            return market_data.get('correlation', 0.0) if market_data else 0.0

        # Calculate stock returns
        stock_returns = df['close'].pct_change().dropna()

        if len(stock_returns) < window:
            return market_data.get('correlation', 0.0) if market_data else 0.0

        if market_data is None:
            return 0.0

        # Try SPY returns first
        spy_prices = market_data.get('spy_prices')
        qqq_prices = market_data.get('qqq_prices')

        correlation = 0.0

        if spy_prices is not None and len(spy_prices) >= len(stock_returns):
            try:
                spy_returns = pd.Series(spy_prices).pct_change().dropna()
                # Align lengths
                min_len = min(len(stock_returns), len(spy_returns))
                if min_len >= window:
                    spy_corr = stock_returns.iloc[-min_len:].corr(spy_returns.iloc[-min_len:])
                    if not np.isnan(spy_corr):
                        correlation = spy_corr
            except Exception:
                pass

        # If SPY didn't work, try QQQ
        if correlation == 0.0 and qqq_prices is not None and len(qqq_prices) >= len(stock_returns):
            try:
                qqq_returns = pd.Series(qqq_prices).pct_change().dropna()
                min_len = min(len(stock_returns), len(qqq_returns))
                if min_len >= window:
                    qqq_corr = stock_returns.iloc[-min_len:].corr(qqq_returns.iloc[-min_len:])
                    if not np.isnan(qqq_corr):
                        correlation = qqq_corr
            except Exception:
                pass

        # Fallback to provided correlation value
        if correlation == 0.0:
            correlation = market_data.get('correlation', 0.0)

        return float(correlation)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 3 time-based features.

        Features:
        1: minutes_since_open
        2: day_of_week
        3: is_option_expiry
        """
        if 'timestamp' not in df.columns:
            # Set default values if timestamp is not available
            df['minutes_since_open'] = 0
            df['day_of_week'] = 0
            df['is_option_expiry'] = 0
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 1: Minutes since market open
        df['minutes_since_open'] = df['timestamp'].apply(self._calculate_minutes_since_open)

        # 2: Day of week (0 = Monday, 4 = Friday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # 3: Option expiry day
        df['is_option_expiry'] = df['timestamp'].dt.date.apply(
            lambda x: 1 if x in self.option_expiry_dates else 0
        )

        return df

    def _calculate_minutes_since_open(self, timestamp: datetime) -> int:
        """Calculate minutes since market open."""
        market_open_today = datetime.combine(timestamp.date(), self.market_open)

        if timestamp >= market_open_today:
            delta = timestamp - market_open_today
            return int(delta.total_seconds() / 60)
        else:
            return 0

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values gracefully.

        Strategy:
        1. Forward-fill for time-series continuity
        2. Backward-fill for remaining NaN at start
        3. Fill any remaining with 0
        """
        # Get feature columns (exclude raw data columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Forward fill
        df[feature_cols] = df[feature_cols].fillna(method='ffill')

        # Backward fill
        df[feature_cols] = df[feature_cols].fillna(method='bfill')

        # Fill any remaining with 0
        df[feature_cols] = df[feature_cols].fillna(0)

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of all 49 feature names.

        Returns:
            List of feature names in order
        """
        features = [
            # Price-based (15)
            'returns_1m', 'returns_5m', 'returns_15m', 'returns_30m', 'returns_60m',
            'ma_5', 'ma_15', 'ma_60',
            'price_vs_ma_5', 'price_vs_ma_15', 'price_vs_ma_60',
            'ma_cross_5_15', 'price_vs_vwap',
            'price_momentum_5m', 'price_momentum_15m',

            # Volatility-based (10)
            'atr_14_normalized', 'bb_position', 'bb_width',
            'volatility_5m', 'volatility_15m', 'volatility_60m',
            'price_acceleration', 'high_low_range', 'intraday_range', 'volatility_ratio',

            # Volume-based (8)
            'volume_ratio', 'volume_ma_5', 'volume_ma_15', 'volume_trend',
            'obv_normalized', 'money_flow_ratio', 'mfi_14', 'vpt_cumsum',

            # Momentum (8)
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'williams_r', 'cci_14',

            # Market context (5)
            'spy_return', 'qqq_return', 'vix_level', 'sector_etf_return', 'market_correlation',

            # Time-based (3)
            'minutes_since_open', 'day_of_week', 'is_option_expiry'
        ]

        return features

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get features grouped by category for analysis.

        Returns:
            Dictionary mapping category names to feature lists
        """
        return {
            'price': [
                'returns_1m', 'returns_5m', 'returns_15m', 'returns_30m', 'returns_60m',
                'ma_5', 'ma_15', 'ma_60',
                'price_vs_ma_5', 'price_vs_ma_15', 'price_vs_ma_60',
                'ma_cross_5_15', 'price_vs_vwap',
                'price_momentum_5m', 'price_momentum_15m'
            ],
            'volatility': [
                'atr_14_normalized', 'bb_position', 'bb_width',
                'volatility_5m', 'volatility_15m', 'volatility_60m',
                'price_acceleration', 'high_low_range', 'intraday_range', 'volatility_ratio'
            ],
            'volume': [
                'volume_ratio', 'volume_ma_5', 'volume_ma_15', 'volume_trend',
                'obv_normalized', 'money_flow_ratio', 'mfi_14', 'vpt_cumsum'
            ],
            'order_book': [
                'bid_ask_spread', 'spread_pct', 'imbalance',
                'bid_depth', 'ask_depth', 'depth_ratio',
                'depth_weighted_mid_price', 'order_flow_imbalance'
            ],
            'momentum': [
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'stoch_k', 'stoch_d', 'williams_r', 'cci_14'
            ],
            'market_context': [
                'spy_return', 'qqq_return', 'vix_level', 'sector_etf_return', 'market_correlation'
            ],
            'time': [
                'minutes_since_open', 'day_of_week', 'is_option_expiry'
            ]
        }
