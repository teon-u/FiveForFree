"""
Test Suite for IncrementalTrainer

Tests cover all major functionality including:
- Initialization
- Sample and batch addition
- Buffer management (FIFO)
- Update conditions
- Incremental training
- Performance tracking
- Rollback capability
- Error recovery
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List

# Import the class to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer.incremental import IncrementalTrainer
from src.models.model_manager import ModelManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager for testing."""
    manager = Mock(spec=ModelManager)
    return manager


@pytest.fixture
def mock_model():
    """Create a mock model with common methods."""
    model = Mock()
    model.is_trained = True
    model.train = Mock()
    model.incremental_train = Mock()
    model.get_recent_accuracy = Mock(return_value=0.75)
    return model


@pytest.fixture
def incremental_trainer(mock_model_manager):
    """Create IncrementalTrainer instance with default settings."""
    return IncrementalTrainer(
        model_manager=mock_model_manager,
        min_samples_for_update=50,
        max_buffer_size=1000
    )


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    X = np.random.randn(100, 20)  # 100 samples, 20 features
    y_up = np.random.randint(0, 2, 100)
    y_down = np.random.randint(0, 2, 100)
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(100)]
    return X, y_up, y_down, timestamps


# ============================================================================
# Test 1: Initialization
# ============================================================================

def test_initialization_default_params(mock_model_manager):
    """Test IncrementalTrainer initialization with default parameters."""
    trainer = IncrementalTrainer(
        model_manager=mock_model_manager,
        min_samples_for_update=50,
        max_buffer_size=1000
    )

    assert trainer.model_manager == mock_model_manager
    assert trainer.min_samples == 50
    assert trainer.max_buffer_size == 1000
    assert trainer.data_buffers == {}
    assert trainer.training_log == []
    assert trainer.performance_history == {}


def test_initialization_custom_params(mock_model_manager):
    """Test IncrementalTrainer initialization with custom parameters."""
    trainer = IncrementalTrainer(
        model_manager=mock_model_manager,
        min_samples_for_update=100,
        max_buffer_size=500
    )

    assert trainer.min_samples == 100
    assert trainer.max_buffer_size == 500


# ============================================================================
# Test 2: Sample Addition
# ============================================================================

def test_add_sample_single(incremental_trainer):
    """Test adding a single sample to the buffer."""
    ticker = "AAPL"
    X = np.random.randn(20)
    y_up = 1.0
    y_down = 0.0
    timestamp = datetime.now()

    incremental_trainer.add_sample(ticker, X, y_up, y_down, timestamp)

    assert ticker in incremental_trainer.data_buffers
    buffer = incremental_trainer.data_buffers[ticker]
    assert len(buffer['X']) == 1
    assert len(buffer['y_up']) == 1
    assert len(buffer['y_down']) == 1
    assert len(buffer['timestamps']) == 1
    assert np.array_equal(buffer['X'][0], X)
    assert buffer['y_up'][0] == y_up
    assert buffer['y_down'][0] == y_down
    assert buffer['timestamps'][0] == timestamp


def test_add_sample_multiple(incremental_trainer):
    """Test adding multiple samples to the buffer."""
    ticker = "AAPL"

    for i in range(10):
        X = np.random.randn(20)
        y_up = float(i % 2)
        y_down = float((i + 1) % 2)
        incremental_trainer.add_sample(ticker, X, y_up, y_down)

    buffer = incremental_trainer.data_buffers[ticker]
    assert len(buffer['X']) == 10
    assert len(buffer['y_up']) == 10
    assert len(buffer['y_down']) == 10
    assert len(buffer['timestamps']) == 10


def test_add_sample_without_timestamp(incremental_trainer):
    """Test adding sample without explicit timestamp (should use current time)."""
    ticker = "AAPL"
    X = np.random.randn(20)
    y_up = 1.0
    y_down = 0.0

    before = datetime.now()
    incremental_trainer.add_sample(ticker, X, y_up, y_down)
    after = datetime.now()

    buffer = incremental_trainer.data_buffers[ticker]
    timestamp = buffer['timestamps'][0]

    assert before <= timestamp <= after


# ============================================================================
# Test 3: Batch Addition
# ============================================================================

def test_add_batch(incremental_trainer, sample_data):
    """Test adding a batch of samples."""
    ticker = "GOOGL"
    X, y_up, y_down, timestamps = sample_data

    incremental_trainer.add_batch(ticker, X, y_up, y_down, timestamps)

    buffer = incremental_trainer.data_buffers[ticker]
    assert len(buffer['X']) == 100
    assert len(buffer['y_up']) == 100
    assert len(buffer['y_down']) == 100
    assert len(buffer['timestamps']) == 100


def test_add_batch_without_timestamps(incremental_trainer):
    """Test adding batch without timestamps."""
    ticker = "MSFT"
    X = np.random.randn(50, 20)
    y_up = np.random.randint(0, 2, 50)
    y_down = np.random.randint(0, 2, 50)

    incremental_trainer.add_batch(ticker, X, y_up, y_down)

    buffer = incremental_trainer.data_buffers[ticker]
    assert len(buffer['X']) == 50
    assert all(isinstance(ts, datetime) for ts in buffer['timestamps'])


def test_add_batch_multiple_tickers(incremental_trainer):
    """Test adding batches for multiple tickers."""
    tickers = ["AAPL", "GOOGL", "MSFT"]

    for ticker in tickers:
        X = np.random.randn(30, 20)
        y_up = np.random.randint(0, 2, 30)
        y_down = np.random.randint(0, 2, 30)
        incremental_trainer.add_batch(ticker, X, y_up, y_down)

    assert len(incremental_trainer.data_buffers) == 3
    for ticker in tickers:
        assert ticker in incremental_trainer.data_buffers
        assert len(incremental_trainer.data_buffers[ticker]['X']) == 30


# ============================================================================
# Test 4: Buffer Size Limit (FIFO)
# ============================================================================

def test_buffer_fifo_behavior(incremental_trainer):
    """Test that buffer maintains FIFO behavior when exceeding max size."""
    ticker = "AAPL"
    max_size = incremental_trainer.max_buffer_size

    # Add more samples than max_buffer_size
    for i in range(max_size + 100):
        X = np.full(20, i)  # Use i as value for tracking
        y_up = float(i)
        y_down = float(i)
        timestamp = datetime.now() + timedelta(seconds=i)
        incremental_trainer.add_sample(ticker, X, y_up, y_down, timestamp)

    buffer = incremental_trainer.data_buffers[ticker]

    # Should only keep max_buffer_size samples
    assert len(buffer['X']) == max_size

    # First sample should be the 100th added (0-99 were removed)
    assert buffer['X'][0][0] == 100

    # Last sample should be the last added
    assert buffer['X'][-1][0] == max_size + 99


def test_buffer_fifo_maintains_chronological_order(incremental_trainer):
    """Test that FIFO maintains chronological order of timestamps."""
    ticker = "AAPL"
    max_size = incremental_trainer.max_buffer_size

    # Add samples with explicit timestamps
    base_time = datetime.now()
    for i in range(max_size + 50):
        X = np.random.randn(20)
        y_up = 1.0
        y_down = 0.0
        timestamp = base_time + timedelta(hours=i)
        incremental_trainer.add_sample(ticker, X, y_up, y_down, timestamp)

    buffer = incremental_trainer.data_buffers[ticker]
    timestamps = buffer['timestamps']

    # Verify chronological order
    for i in range(len(timestamps) - 1):
        assert timestamps[i] < timestamps[i + 1]


# ============================================================================
# Test 5: Update Condition Check
# ============================================================================

def test_should_update_insufficient_samples(incremental_trainer):
    """Test that update is skipped when samples are insufficient."""
    ticker = "AAPL"

    # Add fewer samples than min_samples
    for i in range(incremental_trainer.min_samples - 10):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    results = incremental_trainer.update_models(ticker)

    # Should return empty dict (no update performed)
    assert results == {}


def test_should_update_sufficient_samples(incremental_trainer, mock_model):
    """Test that update proceeds when sufficient samples are available."""
    ticker = "AAPL"

    # Mock the model manager to return a trained model
    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add sufficient samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    results = incremental_trainer.update_models(ticker)

    # Should have results (update attempted)
    assert len(results) > 0


def test_force_update_ignores_min_samples(incremental_trainer, mock_model):
    """Test that force=True bypasses min_samples requirement."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add only 1 sample (way below min_samples)
    X = np.random.randn(20)
    incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    results = incremental_trainer.update_models(ticker, force=True)

    # Should still attempt update
    assert len(results) > 0


# ============================================================================
# Test 6: Incremental Update Execution
# ============================================================================

def test_perform_incremental_update(incremental_trainer, mock_model):
    """Test successful incremental update execution."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    results = incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Verify incremental_train was called
    assert mock_model.incremental_train.called

    # Verify results
    assert "xgboost_up" in results
    assert results["xgboost_up"] is True


def test_update_multiple_targets(incremental_trainer, mock_model):
    """Test updating multiple targets (up and down)."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 1.0)

    results = incremental_trainer.update_models(
        ticker,
        targets=["up", "down"],
        model_types=["xgboost"]
    )

    # Should have results for both targets
    assert "xgboost_up" in results
    assert "xgboost_down" in results


def test_update_multiple_model_types(incremental_trainer, mock_model):
    """Test updating multiple model types."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    results = incremental_trainer.update_models(
        ticker,
        targets=["up"],
        model_types=["xgboost", "lightgbm"]
    )

    # Should have results for both model types
    assert "xgboost_up" in results
    assert "lightgbm_up" in results


# ============================================================================
# Test 7: Performance History Tracking
# ============================================================================

def test_performance_history_recorded(incremental_trainer, mock_model):
    """Test that performance changes are recorded in history."""
    ticker = "AAPL"

    # Mock accuracy changes
    mock_model.get_recent_accuracy = Mock(side_effect=[0.70, 0.75])

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples and update
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Check performance history
    history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")

    assert len(history) > 0
    latest = history[-1]
    assert latest['accuracy_before'] == 0.70
    assert latest['accuracy_after'] == 0.75
    assert latest['improvement'] == pytest.approx(0.05)
    assert latest['n_samples'] == incremental_trainer.min_samples


def test_performance_history_multiple_updates(incremental_trainer, mock_model):
    """Test that multiple updates accumulate in performance history."""
    ticker = "AAPL"

    mock_model.get_recent_accuracy = Mock(return_value=0.75)

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Perform 3 updates
    for update_num in range(3):
        for i in range(incremental_trainer.min_samples):
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

        incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Check history has 3 entries
    history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")
    assert len(history) == 3


def test_get_training_summary(incremental_trainer, mock_model):
    """Test training summary generation."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Perform update
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Get summary
    summary = incremental_trainer.get_training_summary()

    assert summary['total_updates'] == 1
    assert summary['tickers_updated'] == 1
    assert summary['total_samples_processed'] == incremental_trainer.min_samples
    assert 'total_time_seconds' in summary
    assert 'avg_time_per_update' in summary


# ============================================================================
# Test 8: Rollback on Performance Degradation
# ============================================================================

def test_detect_performance_degradation(incremental_trainer, mock_model):
    """Test detection of performance degradation after update."""
    ticker = "AAPL"

    # Mock accuracy degradation
    mock_model.get_recent_accuracy = Mock(side_effect=[0.80, 0.65])

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples and update
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Check that degradation was recorded
    history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")
    latest = history[-1]

    assert latest['improvement'] < 0  # Negative improvement indicates degradation
    assert latest['accuracy_before'] > latest['accuracy_after']


def test_performance_improvement_tracking(incremental_trainer, mock_model):
    """Test tracking of performance improvements."""
    ticker = "AAPL"

    # Mock accuracy improvement
    mock_model.get_recent_accuracy = Mock(side_effect=[0.70, 0.82])

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples and update
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Check improvement
    history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")
    latest = history[-1]

    assert latest['improvement'] > 0
    assert latest['improvement'] == pytest.approx(0.12)


# ============================================================================
# Test 9: Timestamp Management
# ============================================================================

def test_timestamp_tracking(incremental_trainer):
    """Test that timestamps are correctly tracked for samples."""
    ticker = "AAPL"

    timestamps = [
        datetime(2024, 1, 1, 10, 0, 0),
        datetime(2024, 1, 1, 11, 0, 0),
        datetime(2024, 1, 1, 12, 0, 0),
    ]

    for ts in timestamps:
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0, ts)

    buffer = incremental_trainer.data_buffers[ticker]

    assert buffer['timestamps'] == timestamps


def test_get_buffer_status_with_timestamps(incremental_trainer):
    """Test buffer status includes timestamp information."""
    ticker = "AAPL"

    oldest = datetime(2024, 1, 1, 10, 0, 0)
    newest = datetime(2024, 1, 1, 15, 0, 0)

    # Add samples with specific timestamps
    for hour in range(6):
        ts = datetime(2024, 1, 1, 10 + hour, 0, 0)
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0, ts)

    status = incremental_trainer.get_buffer_status(ticker)

    assert status['oldest_sample'] == oldest
    assert status['newest_sample'] == newest


def test_cleanup_old_samples(incremental_trainer):
    """Test cleanup of samples older than specified days."""
    ticker = "AAPL"

    now = datetime.now()

    # Add old samples (10 days ago)
    for i in range(20):
        old_ts = now - timedelta(days=10, hours=i)
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0, old_ts)

    # Add recent samples (2 days ago)
    for i in range(30):
        recent_ts = now - timedelta(days=2, hours=i)
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0, recent_ts)

    # Cleanup samples older than 7 days
    incremental_trainer.cleanup_old_samples(days=7)

    buffer = incremental_trainer.data_buffers[ticker]

    # Should only have recent samples (30)
    assert len(buffer['X']) == 30


# ============================================================================
# Test 10: Multiple Tickers Management
# ============================================================================

def test_multiple_tickers_independent_buffers(incremental_trainer):
    """Test that multiple tickers maintain independent buffers."""
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]

    for idx, ticker in enumerate(tickers):
        for i in range((idx + 1) * 10):  # Different sizes
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Verify each ticker has its own buffer with correct size
    assert len(incremental_trainer.data_buffers) == 4
    assert len(incremental_trainer.data_buffers["AAPL"]['X']) == 10
    assert len(incremental_trainer.data_buffers["GOOGL"]['X']) == 20
    assert len(incremental_trainer.data_buffers["MSFT"]['X']) == 30
    assert len(incremental_trainer.data_buffers["AMZN"]['X']) == 40


def test_update_all_tickers(incremental_trainer, mock_model):
    """Test updating models for all tickers with buffered data."""
    tickers = ["AAPL", "GOOGL", "MSFT"]

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples for each ticker
    for ticker in tickers:
        for i in range(incremental_trainer.min_samples):
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Update all
    results = incremental_trainer.update_all_tickers(
        targets=["up"],
        model_types=["xgboost"]
    )

    # Should have results for all tickers
    assert len(results) == 3
    for ticker in tickers:
        assert ticker in results


def test_get_buffer_status_all_tickers(incremental_trainer):
    """Test getting buffer status for all tickers."""
    tickers = ["AAPL", "GOOGL", "MSFT"]

    for ticker in tickers:
        for i in range(25):  # Below min_samples
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    status = incremental_trainer.get_buffer_status()

    assert status['total_tickers'] == 3
    assert status['ready_for_update'] == 0  # None have enough samples
    assert status['total_samples'] == 75
    assert len(status['tickers']) == 3


# ============================================================================
# Test 11: Data Buffer Clearing
# ============================================================================

def test_clear_buffer_single_ticker(incremental_trainer):
    """Test clearing buffer for a single ticker."""
    ticker = "AAPL"

    # Add samples
    for i in range(50):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    assert ticker in incremental_trainer.data_buffers

    # Clear buffer
    incremental_trainer.clear_buffer(ticker)

    assert ticker not in incremental_trainer.data_buffers


def test_clear_all_buffers(incremental_trainer):
    """Test clearing all buffers."""
    tickers = ["AAPL", "GOOGL", "MSFT"]

    # Add samples for each ticker
    for ticker in tickers:
        for i in range(30):
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    assert len(incremental_trainer.data_buffers) == 3

    # Clear all
    incremental_trainer.clear_all_buffers()

    assert len(incremental_trainer.data_buffers) == 0


def test_buffer_cleared_after_update(incremental_trainer, mock_model):
    """Test that buffer is automatically cleared after successful update."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    assert ticker in incremental_trainer.data_buffers

    # Update
    incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Buffer should be cleared
    assert ticker not in incremental_trainer.data_buffers


# ============================================================================
# Test 12: Long-term Operation Simulation
# ============================================================================

def test_long_term_operation_simulation(incremental_trainer, mock_model):
    """Test simulating long-term incremental learning operation."""
    ticker = "AAPL"

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Simulate 10 days of hourly data additions and periodic updates
    base_time = datetime.now()
    update_count = 0

    for day in range(10):
        for hour in range(24):
            # Add hourly sample
            ts = base_time + timedelta(days=day, hours=hour)
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0, ts)

            # Update every 50 samples (approximately every 2 days)
            buffer_size = len(incremental_trainer.data_buffers.get(ticker, {}).get('X', []))
            if buffer_size >= incremental_trainer.min_samples:
                incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])
                update_count += 1

    # Should have performed multiple updates
    assert update_count > 0

    # Check training log
    assert len(incremental_trainer.training_log) == update_count


def test_continuous_learning_performance_tracking(incremental_trainer, mock_model):
    """Test performance tracking over continuous learning cycles."""
    ticker = "AAPL"

    # Simulate gradual accuracy improvement
    accuracies = [0.65, 0.68, 0.70, 0.73, 0.75, 0.78]
    accuracy_iter = iter(accuracies)

    def get_next_accuracies():
        try:
            acc = next(accuracy_iter)
            return [acc, acc + 0.02]  # before, after
        except StopIteration:
            return [0.78, 0.80]

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Perform 3 update cycles
    for cycle in range(3):
        mock_model.get_recent_accuracy = Mock(side_effect=get_next_accuracies())

        for i in range(incremental_trainer.min_samples):
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

        incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Check performance history shows improvement trend
    history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")
    assert len(history) == 3

    # All updates should show improvement
    for record in history:
        assert record['improvement'] >= 0


# ============================================================================
# Test 13: Error Recovery
# ============================================================================

def test_error_during_model_update(incremental_trainer, mock_model):
    """Test error handling during model update."""
    ticker = "AAPL"

    # Mock model to raise exception during incremental training
    mock_model.incremental_train = Mock(side_effect=Exception("Training failed"))

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Update should handle exception gracefully
    results = incremental_trainer.update_models(ticker, targets=["up"], model_types=["xgboost"])

    # Should return False for failed update
    assert "xgboost_up" in results
    assert results["xgboost_up"] is False


def test_partial_update_failure(incremental_trainer, mock_model):
    """Test handling of partial failures when updating multiple models."""
    ticker = "AAPL"

    # Create two models: one that succeeds, one that fails
    mock_model_success = Mock()
    mock_model_success.is_trained = True
    mock_model_success.incremental_train = Mock()
    mock_model_success.get_recent_accuracy = Mock(return_value=0.75)

    mock_model_failure = Mock()
    mock_model_failure.is_trained = True
    mock_model_failure.incremental_train = Mock(side_effect=Exception("Training failed"))
    mock_model_failure.get_recent_accuracy = Mock(return_value=0.75)

    # Alternate between success and failure models
    def get_model_side_effect(ticker, model_type, target):
        if model_type == "xgboost":
            return (None, mock_model_success)
        else:
            return (None, mock_model_failure)

    incremental_trainer.model_manager.get_or_create_model = Mock(
        side_effect=get_model_side_effect
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Update with both model types
    results = incremental_trainer.update_models(
        ticker,
        targets=["up"],
        model_types=["xgboost", "lightgbm"]
    )

    # Should have mixed results
    assert results["xgboost_up"] is True
    assert results["lightgbm_up"] is False


def test_untrained_model_handling(incremental_trainer, mock_model):
    """Test handling of untrained models during incremental update."""
    ticker = "AAPL"

    # Mock untrained model
    mock_model.is_trained = False
    mock_model.train = Mock()

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )
    incremental_trainer.model_manager.save_model = Mock()

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Update with train_untrained=True
    results = incremental_trainer.update_models(
        ticker,
        targets=["up"],
        model_types=["xgboost"],
        train_untrained=True
    )

    # Should perform initial training
    assert mock_model.train.called
    assert results["xgboost_up"] is True


def test_untrained_model_skip(incremental_trainer, mock_model):
    """Test skipping untrained models when train_untrained=False."""
    ticker = "AAPL"

    # Mock untrained model
    mock_model.is_trained = False

    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Add samples
    for i in range(incremental_trainer.min_samples):
        X = np.random.randn(20)
        incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    # Update with train_untrained=False
    results = incremental_trainer.update_models(
        ticker,
        targets=["up"],
        model_types=["xgboost"],
        train_untrained=False
    )

    # Should skip training
    assert results["xgboost_up"] is False


# ============================================================================
# Test 14: Integration Test (Full Pipeline)
# ============================================================================

def test_full_pipeline_integration(incremental_trainer, mock_model):
    """Test complete incremental learning pipeline from start to finish."""
    tickers = ["AAPL", "GOOGL"]

    # Mock model manager
    incremental_trainer.model_manager.get_or_create_model = Mock(
        return_value=(None, mock_model)
    )

    # Step 1: Add initial batches for multiple tickers
    for ticker in tickers:
        X = np.random.randn(incremental_trainer.min_samples, 20)
        y_up = np.random.randint(0, 2, incremental_trainer.min_samples)
        y_down = np.random.randint(0, 2, incremental_trainer.min_samples)
        incremental_trainer.add_batch(ticker, X, y_up, y_down)

    # Step 2: Check buffer status
    status = incremental_trainer.get_buffer_status()
    assert status['total_tickers'] == 2
    assert status['ready_for_update'] == 2

    # Step 3: Update all tickers
    update_results = incremental_trainer.update_all_tickers(
        targets=["up", "down"],
        model_types=["xgboost", "lightgbm"]
    )

    # Verify updates
    assert len(update_results) == 2
    for ticker in tickers:
        assert ticker in update_results
        assert len(update_results[ticker]) == 4  # 2 targets * 2 models

    # Step 4: Verify buffers cleared after update
    for ticker in tickers:
        assert ticker not in incremental_trainer.data_buffers

    # Step 5: Check training summary
    summary = incremental_trainer.get_training_summary()
    assert summary['total_updates'] == 2
    assert summary['tickers_updated'] == 2

    # Step 6: Add new samples and update again
    for ticker in tickers:
        for i in range(incremental_trainer.min_samples):
            X = np.random.randn(20)
            incremental_trainer.add_sample(ticker, X, 1.0, 0.0)

    update_results_2 = incremental_trainer.update_all_tickers(
        targets=["up"],
        model_types=["xgboost"]
    )

    # Step 7: Verify second round of updates
    assert len(update_results_2) == 2

    # Step 8: Check performance history
    for ticker in tickers:
        history = incremental_trainer.get_performance_history(ticker, "xgboost", "up")
        assert len(history) == 2  # Two updates for each ticker

    # Step 9: Final summary
    final_summary = incremental_trainer.get_training_summary()
    assert final_summary['total_updates'] == 4  # 2 tickers * 2 rounds


def test_export_buffer_to_dataframe(incremental_trainer):
    """Test exporting buffer data to pandas DataFrame."""
    ticker = "AAPL"

    # Add samples
    n_samples = 50
    n_features = 20

    for i in range(n_samples):
        X = np.random.randn(n_features)
        y_up = float(i % 2)
        y_down = float((i + 1) % 2)
        ts = datetime.now() + timedelta(hours=i)
        incremental_trainer.add_sample(ticker, X, y_up, y_down, ts)

    # Export to DataFrame
    df = incremental_trainer.export_buffer_to_dataframe(ticker)

    assert df is not None
    assert len(df) == n_samples
    assert 'timestamp' in df.columns
    assert 'y_up' in df.columns
    assert 'y_down' in df.columns

    # Check feature columns
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    assert len(feature_cols) == n_features


def test_export_buffer_nonexistent_ticker(incremental_trainer):
    """Test exporting buffer for non-existent ticker returns None."""
    df = incremental_trainer.export_buffer_to_dataframe("NONEXISTENT")
    assert df is None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
