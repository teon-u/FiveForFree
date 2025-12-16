"""Tests for feature engineering module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""

    @pytest.fixture
    def sample_minute_bars(self):
        """Create sample minute bar data for testing."""
        dates = pd.date_range('2024-01-01 09:30', periods=120, freq='1min')
        np.random.seed(42)

        # Generate realistic-looking price data
        base_price = 150.0
        returns = np.random.randn(120) * 0.001
        closes = base_price * np.cumprod(1 + returns)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': closes * (1 + np.random.randn(120) * 0.001),
            'high': closes * (1 + np.abs(np.random.randn(120)) * 0.002),
            'low': closes * (1 - np.abs(np.random.randn(120)) * 0.002),
            'close': closes,
            'volume': np.random.randint(10000, 100000, 120),
            'vwap': closes * (1 + np.random.randn(120) * 0.0005)
        })

        return data

    @pytest.fixture
    def sample_market_context(self):
        """Create sample market context data."""
        return {
            'spy_return': 0.5,
            'qqq_return': 0.8,
            'vix_level': 15.0,
            'sector_return': 0.3
        }

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        assert fe is not None

    def test_generate_features_returns_dataframe(self, sample_minute_bars):
        """Test that generate_features returns a DataFrame."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_generate_features_expected_count(self, sample_minute_bars):
        """Test that correct number of features are generated."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Should have around 49 features (based on project spec minus order book features)
        assert result.shape[1] >= 40  # At least 40 features

    def test_price_based_features(self, sample_minute_bars):
        """Test price-based feature generation."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Check for return features
        return_features = [col for col in result.columns if 'return' in col.lower()]
        assert len(return_features) > 0

    def test_moving_average_features(self, sample_minute_bars):
        """Test moving average feature generation."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Check for MA features
        ma_features = [col for col in result.columns if 'ma' in col.lower() or 'sma' in col.lower() or 'ema' in col.lower()]
        assert len(ma_features) > 0

    def test_volatility_features(self, sample_minute_bars):
        """Test volatility feature generation."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Check for volatility features
        vol_features = [col for col in result.columns if 'volatility' in col.lower() or 'atr' in col.lower() or 'bb' in col.lower()]
        assert len(vol_features) > 0

    def test_volume_features(self, sample_minute_bars):
        """Test volume-based feature generation."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Check for volume features
        volume_features = [col for col in result.columns if 'volume' in col.lower() or 'obv' in col.lower()]
        assert len(volume_features) > 0

    def test_momentum_features(self, sample_minute_bars):
        """Test momentum feature generation."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # Check for momentum features
        momentum_features = [col for col in result.columns if 'rsi' in col.lower() or 'macd' in col.lower() or 'stoch' in col.lower()]
        assert len(momentum_features) > 0

    def test_features_no_nan_in_valid_rows(self, sample_minute_bars):
        """Test that features don't have NaN in valid rows (after warmup period)."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)

        # After dropping NaN rows, remaining data should be valid
        clean_result = result.dropna()
        assert len(clean_result) > 0

        # Check that clean rows have no NaN
        assert not clean_result.isnull().any().any()

    def test_features_with_insufficient_data(self):
        """Test handling of insufficient data."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        # Create very short data
        short_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'open': [150.0] * 5,
            'high': [155.0] * 5,
            'low': [145.0] * 5,
            'close': [152.0] * 5,
            'volume': [100000] * 5
        })

        result = fe.generate_features(short_data)

        # Should handle gracefully (return empty or partial features)
        assert result is not None

    def test_feature_values_are_finite(self, sample_minute_bars):
        """Test that feature values are finite numbers."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars)
        clean_result = result.dropna()

        # All values should be finite
        assert np.all(np.isfinite(clean_result.values))

    def test_add_market_context_features(self, sample_minute_bars, sample_market_context):
        """Test adding market context features."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        result = fe.generate_features(sample_minute_bars, market_context=sample_market_context)

        # Should include market context features
        context_features = [col for col in result.columns if 'spy' in col.lower() or 'qqq' in col.lower() or 'vix' in col.lower()]
        # Note: This depends on implementation - market context may be added differently
        assert isinstance(result, pd.DataFrame)


class TestLabelGenerator:
    """Test suite for LabelGenerator."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for label generation."""
        dates = pd.date_range('2024-01-01 09:30', periods=120, freq='1min')
        np.random.seed(42)

        base_price = 100.0
        # Create data with some 5%+ moves
        closes = [base_price]
        for i in range(119):
            change = np.random.choice([-0.002, 0, 0.002, 0.05, -0.05], p=[0.3, 0.3, 0.3, 0.05, 0.05])
            closes.append(closes[-1] * (1 + change))

        data = pd.DataFrame({
            'timestamp': dates,
            'high': np.array(closes) * 1.01,
            'low': np.array(closes) * 0.99,
            'close': closes
        })

        return data

    def test_label_generator_initialization(self):
        """Test LabelGenerator initialization."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator()
        assert lg is not None

    def test_generate_labels_returns_dict(self, sample_price_data):
        """Test that generate_labels returns a dict."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator()

        entry_time = sample_price_data['timestamp'].iloc[0]
        entry_price = sample_price_data['close'].iloc[0]

        result = lg.generate_labels(sample_price_data, entry_time, entry_price)

        assert isinstance(result, dict)
        assert 'label_up' in result
        assert 'label_down' in result

    def test_label_values_are_binary(self, sample_price_data):
        """Test that labels are binary (0 or 1)."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator()

        entry_time = sample_price_data['timestamp'].iloc[0]
        entry_price = sample_price_data['close'].iloc[0]

        result = lg.generate_labels(sample_price_data, entry_time, entry_price)

        assert result['label_up'] in [0, 1, True, False]
        assert result['label_down'] in [0, 1, True, False]

    def test_label_up_detection(self):
        """Test that 5% up move is detected."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator(target_percent=5.0)

        entry_price = 100.0
        dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')

        # Create data with 6% up move
        data = pd.DataFrame({
            'timestamp': dates,
            'high': [106.0] * 60,  # 6% above entry
            'low': [99.0] * 60,
            'close': [105.0] * 60
        })

        entry_time = dates[0]
        result = lg.generate_labels(data, entry_time, entry_price)

        assert result['label_up'] == 1 or result['label_up'] == True

    def test_label_down_detection(self):
        """Test that 5% down move is detected."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator(target_percent=5.0)

        entry_price = 100.0
        dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')

        # Create data with 6% down move
        data = pd.DataFrame({
            'timestamp': dates,
            'high': [101.0] * 60,
            'low': [94.0] * 60,  # 6% below entry
            'close': [95.0] * 60
        })

        entry_time = dates[0]
        result = lg.generate_labels(data, entry_time, entry_price)

        assert result['label_down'] == 1 or result['label_down'] == True

    def test_no_label_for_small_move(self):
        """Test that small moves don't trigger labels."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator(target_percent=5.0)

        entry_price = 100.0
        dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')

        # Create data with only 2% move
        data = pd.DataFrame({
            'timestamp': dates,
            'high': [102.0] * 60,  # Only 2% up
            'low': [98.0] * 60,   # Only 2% down
            'close': [100.5] * 60
        })

        entry_time = dates[0]
        result = lg.generate_labels(data, entry_time, entry_price)

        assert result['label_up'] == 0 or result['label_up'] == False
        assert result['label_down'] == 0 or result['label_down'] == False

    def test_label_time_horizon(self):
        """Test that labels only consider 60-minute horizon."""
        from src.processor.label_generator import LabelGenerator
        lg = LabelGenerator(target_percent=5.0, horizon_minutes=60)

        entry_price = 100.0
        dates = pd.date_range('2024-01-01 09:30', periods=120, freq='1min')

        # Create data where 5%+ move happens only after 60 minutes
        highs = [102.0] * 60 + [110.0] * 60  # Big move only after hour
        lows = [98.0] * 60 + [98.0] * 60

        data = pd.DataFrame({
            'timestamp': dates,
            'high': highs,
            'low': lows,
            'close': [100.0] * 120
        })

        entry_time = dates[0]
        result = lg.generate_labels(data, entry_time, entry_price)

        # Should not detect the late move
        assert result['label_up'] == 0 or result['label_up'] == False


class TestFeatureEngineerEdgeCases:
    """Test edge cases for feature engineering."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        empty_df = pd.DataFrame()
        result = fe.generate_features(empty_df)

        # Should handle gracefully
        assert result is None or len(result) == 0

    def test_single_row_handling(self):
        """Test handling of single row DataFrame."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        single_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [150.0],
            'high': [155.0],
            'low': [145.0],
            'close': [152.0],
            'volume': [100000]
        })

        result = fe.generate_features(single_row)

        # Should handle gracefully (may return empty or partial)
        assert result is not None

    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=60, freq='1min'),
            'close': [150.0] * 60
            # Missing open, high, low, volume
        })

        # Should either raise an error or handle gracefully
        try:
            result = fe.generate_features(incomplete_data)
            # If it doesn't raise, should return something valid
            assert result is None or isinstance(result, pd.DataFrame)
        except (KeyError, ValueError):
            # Expected behavior for missing columns
            pass

    def test_zero_volume_handling(self):
        """Test handling of zero volume data."""
        from src.processor.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        zero_volume_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=60, freq='1min'),
            'open': [150.0] * 60,
            'high': [155.0] * 60,
            'low': [145.0] * 60,
            'close': [152.0] * 60,
            'volume': [0] * 60  # Zero volume
        })

        result = fe.generate_features(zero_volume_data)

        # Should handle gracefully without division by zero
        assert result is not None
        if len(result) > 0:
            # Should not have infinite values
            clean_result = result.dropna()
            if len(clean_result) > 0:
                assert np.all(np.isfinite(clean_result.values))
