"""
Feature Engineering Module for NASDAQ Prediction System

Generates 57 features across 7 categories:
- Price-based (15)
- Volatility-based (10)
- Volume-based (8)
- Order book (8)
- Momentum (8)
- Market context (5)
- Time-based (3)
"""

from datetime import datetime, time
from typing import Dict, Optional, List
import warnings

import numpy as np
import pandas as pd
import talib

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for high-frequency NASDAQ trading.

    All features are computed using vectorized operations for performance.
    Handles missing data gracefully with forward-fill and fallback values.
    """

    def __init__(self, market_open: time = time(9, 30), market_close: time = time(16, 0)):
        """
        Initialize feature engineer.

        Args:
            market_open: Market opening time (default 9:30 AM)
            market_close: Market closing time (default 4:00 PM)
        """
        self.market_open = market_open
        self.market_close = market_close

        # Option expiry dates (third Friday of each month)
        self.option_expiry_dates = self._generate_option_expiry_dates()

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
        market_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compute all 57 features from minute bar data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume, vwap]
            order_book: Dict with order book data (optional)
            market_data: Dict with SPY, QQQ, VIX, sector data (optional)

        Returns:
            DataFrame with all computed features
        """
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

        # 4. Order book features (8)
        df = self._add_order_book_features(df, order_book)

        # 5. Momentum features (8)
        df = self._add_momentum_features(df)

        # 6. Market context features (5)
        df = self._add_market_context_features(df, market_data)

        # 7. Time-based features (3)
        df = self._add_time_features(df)

        # Fill any remaining NaN values
        df = self._handle_missing_values(df)

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
        df['ma_5'] = talib.SMA(close, timeperiod=5)
        df['ma_15'] = talib.SMA(close, timeperiod=15)
        df['ma_60'] = talib.SMA(close, timeperiod=60)

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
        df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['atr_14_normalized'] = df['atr_14'] / df['close']

        # 2-3: Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
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
        df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)

        # 2-3: Volume moving averages
        df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
        df['volume_ma_15'] = talib.SMA(volume, timeperiod=15)

        # 4: Volume trend
        df['volume_trend'] = (df['volume_ma_5'] - df['volume_ma_15']) / (df['volume_ma_15'] + 1e-10)

        # 5: On-Balance Volume
        df['obv'] = talib.OBV(close, volume)
        df['obv_normalized'] = df['obv'] / df['obv'].rolling(60).mean()

        # 6: Money flow (price * volume)
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()

        # 7: Money Flow Index
        df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)

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
            # Set default values when order book data is not available
            df['bid_ask_spread'] = 0.0
            df['spread_pct'] = 0.0
            df['imbalance'] = 0.0
            df['bid_depth'] = 0.0
            df['ask_depth'] = 0.0
            df['depth_ratio'] = 1.0
            df['depth_weighted_mid_price'] = df['close']
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
                df['depth_ratio'] = 1.0
                df['depth_weighted_mid_price'] = df['close']
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
        df['rsi_14'] = talib.RSI(close, timeperiod=14)

        # 2-4: MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # 5-6: Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # 7: Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

        # 8: Commodity Channel Index
        df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)

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
            # This would need historical data to compute properly
            df['market_correlation'] = market_data.get('correlation', 0.0)

        return df

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
        Get list of all 57 feature names.

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

            # Order book (8)
            'bid_ask_spread', 'spread_pct', 'imbalance',
            'bid_depth', 'ask_depth', 'depth_ratio',
            'depth_weighted_mid_price', 'order_flow_imbalance',

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
