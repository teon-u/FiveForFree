"""
Demonstration of Feature Engineering and Label Generation

This script shows how to use the FeatureEngineer and LabelGenerator classes
to process NASDAQ market data for prediction.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.processor import FeatureEngineer, LabelGenerator


def generate_sample_data(n_minutes: int = 200) -> pd.DataFrame:
    """Generate sample minute bar data for testing."""
    np.random.seed(42)

    # Generate timestamps (market hours)
    start_time = datetime(2024, 1, 15, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_minutes)]

    # Generate realistic price data with trend and noise
    base_price = 150.0
    trend = np.cumsum(np.random.randn(n_minutes) * 0.5)
    prices = base_price + trend

    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        high = close * (1 + abs(np.random.randn() * 0.005))
        low = close * (1 - abs(np.random.randn() * 0.005))
        open_price = prices[i-1] if i > 0 else close

        volume = int(np.random.exponential(100000))

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def demo_feature_engineering():
    """Demonstrate feature engineering."""
    print("=" * 80)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)

    # Generate sample data
    print("\n1. Generating sample minute bar data...")
    df = generate_sample_data(n_minutes=200)
    print(f"   Generated {len(df)} minute bars")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    # Initialize feature engineer
    print("\n2. Initializing FeatureEngineer...")
    feature_eng = FeatureEngineer()

    # Sample order book data
    order_book = {
        'bids': [(149.95, 1000), (149.94, 1500), (149.93, 2000)],
        'asks': [(150.05, 800), (150.06, 1200), (150.07, 1800)],
        'bid_total_volume': 4500,
        'ask_total_volume': 3800,
        'imbalance': 0.084
    }

    # Sample market data
    market_data = {
        'spy_return': 0.0025,
        'qqq_return': 0.0032,
        'vix_level': 18.5,
        'sector_return': 0.0028,
        'correlation': 0.75
    }

    # Compute features
    print("\n3. Computing 57 features...")
    df_features = feature_eng.compute_features(df, order_book, market_data)

    # Get feature names
    feature_names = feature_eng.get_feature_names()
    print(f"   Total features: {len(feature_names)}")

    # Show feature groups
    print("\n4. Feature breakdown by category:")
    feature_groups = feature_eng.get_feature_importance_groups()
    for category, features in feature_groups.items():
        print(f"   {category.upper()}: {len(features)} features")

    # Display sample features
    print("\n5. Sample feature values (last row):")
    sample_features = [
        'returns_1m', 'returns_5m', 'ma_5', 'ma_15',
        'rsi_14', 'macd', 'bb_position',
        'volume_ratio', 'mfi_14',
        'bid_ask_spread', 'imbalance',
        'vix_level', 'day_of_week'
    ]

    for feat in sample_features:
        if feat in df_features.columns:
            value = df_features[feat].iloc[-1]
            print(f"   {feat:25s} = {value:12.6f}")

    # Check for missing values
    print("\n6. Data quality check:")
    missing_counts = df_features[feature_names].isnull().sum()
    total_missing = missing_counts.sum()
    print(f"   Total missing values: {total_missing}")
    print(f"   Features with missing values: {(missing_counts > 0).sum()}")

    return df_features


def demo_label_generation():
    """Demonstrate label generation."""
    print("\n" + "=" * 80)
    print("LABEL GENERATION DEMONSTRATION")
    print("=" * 80)

    # Generate sample data with higher volatility
    print("\n1. Generating sample data with volatility...")
    df = generate_sample_data(n_minutes=200)

    # Add some volatility spikes
    for i in [50, 100, 150]:
        if i < len(df):
            df.loc[i:i+10, 'high'] *= 1.06  # 6% spike
            df.loc[i+5:i+15, 'low'] *= 0.94  # 6% dip

    print(f"   Generated {len(df)} minute bars with volatility spikes")

    # Initialize label generator
    print("\n2. Initializing LabelGenerator...")
    label_gen = LabelGenerator(
        target_percent=5.0,
        prediction_horizon_minutes=60,
        commission_pct=0.1
    )

    # Generate labels (vectorized method)
    print("\n3. Generating labels (vectorized method)...")
    df_labels = label_gen.generate_labels_vectorized(df)

    # Get label statistics
    print("\n4. Label statistics:")
    stats = label_gen.get_label_statistics(df_labels)

    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Up signals: {stats['up_count']} ({stats['up_rate']*100:.1f}%)")
    print(f"   Down signals: {stats['down_count']} ({stats['down_rate']*100:.1f}%)")
    print(f"   Both signals: {stats['both_count']} ({stats['both_rate']*100:.1f}%)")
    print(f"   Neither signals: {stats['neither_count']} ({stats['neither_rate']*100:.1f}%)")
    print(f"   Average max gain: {stats['avg_max_gain']:.2f}%")
    print(f"   Average max loss: {stats['avg_max_loss']:.2f}%")

    # Validate labels
    print("\n5. Label validation:")
    validation = label_gen.validate_labels(df_labels)
    print(f"   Is valid: {validation['is_valid']}")
    print(f"   Has label_up: {validation['has_label_up']}")
    print(f"   Has label_down: {validation['has_label_down']}")
    print(f"   Severe up imbalance: {validation['severe_up_imbalance']}")
    print(f"   Severe down imbalance: {validation['severe_down_imbalance']}")

    # Generate labels for specific entry point
    print("\n6. Single entry point example:")
    entry_idx = 100
    entry_time = df.loc[entry_idx, 'timestamp']
    entry_price = df.loc[entry_idx, 'close']

    labels = label_gen.generate_labels(df, entry_time, entry_price)

    print(f"   Entry time: {entry_time}")
    print(f"   Entry price: ${entry_price:.2f}")
    print(f"   Label up: {labels['label_up']}")
    print(f"   Label down: {labels['label_down']}")
    print(f"   Max gain: {labels['max_gain']:.2f}%")
    print(f"   Max loss: {labels['max_loss']:.2f}%")
    if labels['minutes_to_target_up']:
        print(f"   Minutes to up target: {labels['minutes_to_target_up']:.1f}")
    if labels['minutes_to_target_down']:
        print(f"   Minutes to down target: {labels['minutes_to_target_down']:.1f}")

    return df_labels


def demo_integrated_pipeline():
    """Demonstrate integrated feature engineering and label generation."""
    print("\n" + "=" * 80)
    print("INTEGRATED PIPELINE DEMONSTRATION")
    print("=" * 80)

    # Generate sample data
    print("\n1. Generating sample market data...")
    df = generate_sample_data(n_minutes=200)

    # Initialize modules
    print("\n2. Initializing modules...")
    feature_eng = FeatureEngineer()
    label_gen = LabelGenerator(target_percent=5.0, prediction_horizon_minutes=60)

    # Compute features
    print("\n3. Computing features...")
    df_features = feature_eng.compute_features(df)

    # Generate labels
    print("\n4. Generating labels...")
    df_labels = label_gen.generate_labels_vectorized(df)

    # Merge features and labels
    print("\n5. Merging features and labels...")
    feature_cols = feature_eng.get_feature_names()
    label_cols = ['label_up', 'label_down', 'max_gain', 'max_loss']

    # Create training dataset
    X = df_features[feature_cols].copy()
    y_up = df_labels['label_up'].copy()
    y_down = df_labels['label_down'].copy()

    print(f"\n6. Training dataset ready:")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Label up shape: {y_up.shape}")
    print(f"   Label down shape: {y_down.shape}")
    print(f"   Up positive rate: {y_up.mean()*100:.1f}%")
    print(f"   Down positive rate: {y_down.mean()*100:.1f}%")

    # Show sample of feature matrix
    print(f"\n7. Sample feature matrix (first 5 features, last 3 rows):")
    print(X[feature_cols[:5]].tail(3).to_string())

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe feature engineering module successfully generated 57 features")
    print("and the label generator created binary labels for up/down predictions.")
    print("\nThese modules are ready for integration with the model training pipeline.")


if __name__ == '__main__':
    # Run demonstrations
    df_features = demo_feature_engineering()
    df_labels = demo_label_generation()
    demo_integrated_pipeline()
