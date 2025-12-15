"""
Label Generation Module for NASDAQ Prediction System

Generates binary labels for up/down predictions based on:
- 5% threshold for significant price movement
- 60-minute prediction horizon
- Tracks maximum gain/loss for analysis
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class LabelGenerator:
    """
    Generate training labels for NASDAQ prediction models.

    Labels:
    - label_up: 1 if price reaches +5% within 60 minutes, else 0
    - label_down: 1 if price reaches -5% within 60 minutes, else 0

    Both labels can be 1 simultaneously (high volatility scenario).
    """

    def __init__(
        self,
        target_percent: float = 5.0,
        prediction_horizon_minutes: int = 60,
        commission_pct: float = 0.1
    ):
        """
        Initialize label generator.

        Args:
            target_percent: Target profit/loss percentage (default 5.0)
            prediction_horizon_minutes: Time horizon for prediction (default 60)
            commission_pct: Trading commission per trade (default 0.1%)
        """
        self.target_percent = target_percent
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.commission_pct = commission_pct

    def generate_labels(
        self,
        minute_bars: pd.DataFrame,
        entry_time: datetime,
        entry_price: float
    ) -> Dict[str, float]:
        """
        Generate labels for a single entry point.

        Args:
            minute_bars: DataFrame with columns [timestamp, open, high, low, close, volume]
            entry_time: Entry timestamp
            entry_price: Entry price

        Returns:
            Dictionary with:
            - label_up: Binary (1/0) for upward movement
            - label_down: Binary (1/0) for downward movement
            - max_gain: Maximum gain percentage achieved
            - max_loss: Maximum loss percentage achieved
            - exit_price_up: Exit price if target hit (up)
            - exit_price_down: Exit price if target hit (down)
            - minutes_to_target_up: Minutes to reach up target
            - minutes_to_target_down: Minutes to reach down target
        """
        # Get future bars within prediction horizon
        end_time = entry_time + timedelta(minutes=self.prediction_horizon_minutes)

        future_bars = minute_bars[
            (minute_bars['timestamp'] > entry_time) &
            (minute_bars['timestamp'] <= end_time)
        ].copy()

        if len(future_bars) == 0:
            # No future data available
            return self._create_default_labels()

        # Calculate gain/loss percentages
        future_bars['gain_pct'] = ((future_bars['high'] - entry_price) / entry_price) * 100
        future_bars['loss_pct'] = ((future_bars['low'] - entry_price) / entry_price) * 100

        # Account for commission (both entry and exit)
        commission_total = self.commission_pct * 2  # Round-trip commission

        # Adjust for commission
        future_bars['gain_pct_net'] = future_bars['gain_pct'] - commission_total
        future_bars['loss_pct_net'] = future_bars['loss_pct'] + commission_total

        # Find maximum gain and loss
        max_gain = future_bars['gain_pct_net'].max()
        max_loss = future_bars['loss_pct_net'].min()

        # Check if target was reached
        label_up = 1 if max_gain >= self.target_percent else 0
        label_down = 1 if max_loss <= -self.target_percent else 0

        # Find when targets were reached
        up_target_bars = future_bars[future_bars['gain_pct_net'] >= self.target_percent]
        down_target_bars = future_bars[future_bars['loss_pct_net'] <= -self.target_percent]

        # Calculate exit prices and timing
        if len(up_target_bars) > 0:
            first_up_bar = up_target_bars.iloc[0]
            exit_price_up = entry_price * (1 + self.target_percent / 100)
            minutes_to_target_up = (first_up_bar['timestamp'] - entry_time).total_seconds() / 60
        else:
            exit_price_up = None
            minutes_to_target_up = None

        if len(down_target_bars) > 0:
            first_down_bar = down_target_bars.iloc[0]
            exit_price_down = entry_price * (1 - self.target_percent / 100)
            minutes_to_target_down = (first_down_bar['timestamp'] - entry_time).total_seconds() / 60
        else:
            exit_price_down = None
            minutes_to_target_down = None

        return {
            'label_up': label_up,
            'label_down': label_down,
            'max_gain': max_gain,
            'max_loss': max_loss,
            'exit_price_up': exit_price_up,
            'exit_price_down': exit_price_down,
            'minutes_to_target_up': minutes_to_target_up,
            'minutes_to_target_down': minutes_to_target_down,
            'entry_price': entry_price,
            'entry_time': entry_time
        }

    def generate_labels_batch(
        self,
        minute_bars: pd.DataFrame,
        lookback_minutes: int = 120
    ) -> pd.DataFrame:
        """
        Generate labels for all valid entry points in the dataset.

        Args:
            minute_bars: DataFrame with columns [timestamp, open, high, low, close, volume]
            lookback_minutes: Minutes of history required before generating labels

        Returns:
            DataFrame with all labels and metadata
        """
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(minute_bars['timestamp']):
            minute_bars = minute_bars.copy()
            minute_bars['timestamp'] = pd.to_datetime(minute_bars['timestamp'])

        # Sort by timestamp
        minute_bars = minute_bars.sort_values('timestamp').reset_index(drop=True)

        # Generate labels for each valid entry point
        labels_list = []

        for idx in range(lookback_minutes, len(minute_bars)):
            entry_time = minute_bars.loc[idx, 'timestamp']
            entry_price = minute_bars.loc[idx, 'close']

            # Generate labels
            labels = self.generate_labels(minute_bars, entry_time, entry_price)

            # Add index and timestamp
            labels['idx'] = idx
            labels['timestamp'] = entry_time

            labels_list.append(labels)

        # Convert to DataFrame
        labels_df = pd.DataFrame(labels_list)

        return labels_df

    def generate_labels_vectorized(
        self,
        minute_bars: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate labels using vectorized operations for better performance.

        Args:
            minute_bars: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with label_up and label_down columns added
        """
        df = minute_bars.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Initialize label columns
        df['label_up'] = 0
        df['label_down'] = 0
        df['max_gain'] = 0.0
        df['max_loss'] = 0.0

        # Calculate rolling maximum high and minimum low for next N minutes
        horizon = self.prediction_horizon_minutes

        # Use rolling window with reverse indexing
        for i in range(len(df) - horizon):
            entry_price = df.loc[i, 'close']

            # Get future bars
            future_highs = df.loc[i+1:i+horizon, 'high'].values
            future_lows = df.loc[i+1:i+horizon, 'low'].values

            if len(future_highs) == 0:
                continue

            # Calculate max gain and loss
            max_high = np.max(future_highs)
            min_low = np.min(future_lows)

            max_gain = ((max_high - entry_price) / entry_price) * 100
            max_loss = ((min_low - entry_price) / entry_price) * 100

            # Adjust for commission
            commission_total = self.commission_pct * 2
            max_gain_net = max_gain - commission_total
            max_loss_net = max_loss + commission_total

            # Set labels
            df.loc[i, 'label_up'] = 1 if max_gain_net >= self.target_percent else 0
            df.loc[i, 'label_down'] = 1 if max_loss_net <= -self.target_percent else 0
            df.loc[i, 'max_gain'] = max_gain_net
            df.loc[i, 'max_loss'] = max_loss_net

        return df

    def _create_default_labels(self) -> Dict[str, float]:
        """Create default labels when no future data is available."""
        return {
            'label_up': 0,
            'label_down': 0,
            'max_gain': 0.0,
            'max_loss': 0.0,
            'exit_price_up': None,
            'exit_price_down': None,
            'minutes_to_target_up': None,
            'minutes_to_target_down': None,
            'entry_price': None,
            'entry_time': None
        }

    def get_label_statistics(self, labels_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics about generated labels.

        Args:
            labels_df: DataFrame with label columns

        Returns:
            Dictionary with statistics
        """
        total = len(labels_df)

        if total == 0:
            return {
                'total_samples': 0,
                'up_rate': 0.0,
                'down_rate': 0.0,
                'both_rate': 0.0,
                'neither_rate': 0.0,
                'avg_max_gain': 0.0,
                'avg_max_loss': 0.0,
                'avg_time_to_target_up': 0.0,
                'avg_time_to_target_down': 0.0
            }

        up_count = labels_df['label_up'].sum()
        down_count = labels_df['label_down'].sum()
        both_count = ((labels_df['label_up'] == 1) & (labels_df['label_down'] == 1)).sum()
        neither_count = ((labels_df['label_up'] == 0) & (labels_df['label_down'] == 0)).sum()

        # Calculate averages
        avg_max_gain = labels_df['max_gain'].mean()
        avg_max_loss = labels_df['max_loss'].mean()

        # Time to target (only for successful cases)
        if 'minutes_to_target_up' in labels_df.columns:
            avg_time_up = labels_df[labels_df['label_up'] == 1]['minutes_to_target_up'].mean()
            avg_time_down = labels_df[labels_df['label_down'] == 1]['minutes_to_target_down'].mean()
        else:
            avg_time_up = 0.0
            avg_time_down = 0.0

        return {
            'total_samples': total,
            'up_count': int(up_count),
            'down_count': int(down_count),
            'both_count': int(both_count),
            'neither_count': int(neither_count),
            'up_rate': float(up_count / total),
            'down_rate': float(down_count / total),
            'both_rate': float(both_count / total),
            'neither_rate': float(neither_count / total),
            'avg_max_gain': float(avg_max_gain),
            'avg_max_loss': float(avg_max_loss),
            'avg_time_to_target_up': float(avg_time_up) if not np.isnan(avg_time_up) else 0.0,
            'avg_time_to_target_down': float(avg_time_down) if not np.isnan(avg_time_down) else 0.0
        }

    def balance_dataset(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        target: str = 'up',
        method: str = 'undersample'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Balance dataset for training.

        Args:
            features_df: Features DataFrame
            labels_df: Labels DataFrame
            target: 'up' or 'down'
            method: 'undersample' or 'oversample'

        Returns:
            Tuple of (balanced_features, balanced_labels)
        """
        label_col = f'label_{target}'

        if label_col not in labels_df.columns:
            raise ValueError(f"Label column {label_col} not found")

        # Count positive and negative samples
        positive_mask = labels_df[label_col] == 1
        negative_mask = labels_df[label_col] == 0

        n_positive = positive_mask.sum()
        n_negative = negative_mask.sum()

        if method == 'undersample':
            # Undersample the majority class
            if n_positive > n_negative:
                # Too many positive, sample down
                positive_idx = labels_df[positive_mask].sample(n=n_negative, random_state=42).index
                negative_idx = labels_df[negative_mask].index
            else:
                # Too many negative, sample down
                positive_idx = labels_df[positive_mask].index
                negative_idx = labels_df[negative_mask].sample(n=n_positive, random_state=42).index

            balanced_idx = positive_idx.union(negative_idx)

        elif method == 'oversample':
            # Oversample the minority class
            if n_positive < n_negative:
                # Too few positive, sample up
                positive_idx = labels_df[positive_mask].sample(n=n_negative, replace=True, random_state=42).index
                negative_idx = labels_df[negative_mask].index
            else:
                # Too few negative, sample up
                positive_idx = labels_df[positive_mask].index
                negative_idx = labels_df[negative_mask].sample(n=n_positive, replace=True, random_state=42).index

            balanced_idx = positive_idx.union(negative_idx)

        else:
            raise ValueError(f"Unknown balancing method: {method}")

        # Return balanced datasets
        balanced_features = features_df.loc[balanced_idx].reset_index(drop=True)
        balanced_labels = labels_df.loc[balanced_idx].reset_index(drop=True)

        return balanced_features, balanced_labels

    def validate_labels(self, labels_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate label quality and detect potential issues.

        Args:
            labels_df: DataFrame with labels

        Returns:
            Dictionary with validation results
        """
        issues = {}

        # Check for required columns
        required_cols = ['label_up', 'label_down']
        for col in required_cols:
            issues[f'has_{col}'] = col in labels_df.columns

        if not all(issues.values()):
            return issues

        # Check class imbalance
        up_rate = labels_df['label_up'].mean()
        down_rate = labels_df['label_down'].mean()

        issues['severe_up_imbalance'] = up_rate < 0.05 or up_rate > 0.95
        issues['severe_down_imbalance'] = down_rate < 0.05 or down_rate > 0.95

        # Check for data leakage (all samples have same label)
        issues['up_all_same'] = labels_df['label_up'].nunique() == 1
        issues['down_all_same'] = labels_df['label_down'].nunique() == 1

        # Check for missing values
        issues['has_missing'] = labels_df[required_cols].isnull().any().any()

        # Overall validation
        issues['is_valid'] = not any([
            issues['severe_up_imbalance'],
            issues['severe_down_imbalance'],
            issues['up_all_same'],
            issues['down_all_same'],
            issues['has_missing']
        ])

        return issues
