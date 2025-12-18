"""
GPU Trainer Tests for NASDAQ Prediction System

Tests for GPUParallelTrainer class covering:
- Initialization and configuration
- GPU detection and fallback
- Training workflows (single/batch)
- Parallel and sequential execution
- Memory management
- Error handling
- Integration scenarios

Test Structure:
- 16 comprehensive test cases
- Mock-based testing for ModelManager
- Conditional GPU tests (skip if unavailable)
- Realistic data generation with numpy
"""

import gc
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Tuple

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from src.trainer.gpu_trainer import GPUParallelTrainer
    from src.models.model_manager import ModelManager
    GPU_TRAINER_AVAILABLE = True
except ImportError:
    GPU_TRAINER_AVAILABLE = False


# Skip all tests if required modules are not available
pytestmark = pytest.mark.skipif(
    not GPU_TRAINER_AVAILABLE,
    reason="GPUParallelTrainer or dependencies not available"
)


# ========== Test Fixtures ==========

@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager for testing."""
    manager = Mock(spec=ModelManager)

    # Mock model instance
    mock_model = Mock()
    mock_model.train = Mock(return_value=None)
    mock_model.predict_proba = Mock(return_value=np.random.rand(10))
    mock_model.is_trained = True

    # get_or_create_model returns (model_type, model)
    manager.get_or_create_model = Mock(return_value=("xgboost", mock_model))
    manager.save_models = Mock(return_value=None)
    manager.save_model = Mock(return_value=Path("/fake/path/model.pkl"))

    return manager


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 57  # Typical feature count

    X = np.random.randn(n_samples, n_features)
    y_up = np.random.randint(0, 2, n_samples)
    y_down = np.random.randint(0, 2, n_samples)
    y_volatility = np.random.randint(0, 2, n_samples)
    y_direction = np.random.randint(0, 2, n_samples)

    return {
        'X': X,
        'y_up': y_up,
        'y_down': y_down,
        'y_volatility': y_volatility,
        'y_direction': y_direction
    }


@pytest.fixture
def batch_ticker_data(sample_data):
    """Generate batch ticker data for testing."""
    return {
        'AAPL': (sample_data['X'], sample_data['y_up']),
        'GOOGL': (sample_data['X'], sample_data['y_down']),
        'MSFT': (sample_data['X'], sample_data['y_up'])
    }


# ========== Test 1: Initialization ==========

def test_initialization(mock_model_manager):
    """Test 1: GPUParallelTrainer initialization with parameters."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 4
        mock_settings.USE_GPU = True

        trainer = GPUParallelTrainer(
            model_manager=mock_model_manager,
            n_parallel=2,
            validation_split=0.15
        )

        assert trainer.model_manager == mock_model_manager
        assert trainer.n_parallel == 2
        assert trainer.validation_split == 0.15
        assert isinstance(trainer.training_history, dict)
        assert len(trainer.training_history) == 0


def test_initialization_default_params(mock_model_manager):
    """Test 1b: Initialization with default parameters."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 8
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        assert trainer.n_parallel == 8
        assert trainer.validation_split == 0.2  # Default value


# ========== Test 2: GPU Detection ==========

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gpu_detection_available(mock_model_manager):
    """Test 2: GPU detection when CUDA is available."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5080'), \
         patch('torch.cuda.get_device_properties') as mock_props:

        mock_settings.USE_GPU = True
        mock_settings.N_PARALLEL_WORKERS = 4

        # Mock device properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1e9  # 16GB
        mock_props.return_value = mock_device_props

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        assert trainer.use_gpu is True
        assert str(trainer.device) == 'cuda:0'


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gpu_detection_unavailable(mock_model_manager):
    """Test 2b: GPU detection when CUDA is not available."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings, \
         patch('torch.cuda.is_available', return_value=False):

        mock_settings.USE_GPU = True
        mock_settings.N_PARALLEL_WORKERS = 4

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        assert str(trainer.device) == 'cpu'


# ========== Test 3: CPU Fallback ==========

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_cpu_fallback(mock_model_manager):
    """Test 3: CPU fallback when GPU is disabled in settings."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.USE_GPU = False
        mock_settings.N_PARALLEL_WORKERS = 4

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        assert trainer.use_gpu is False
        assert str(trainer.device) == 'cpu'


# ========== Test 4: Single Ticker Training ==========

def test_single_ticker_training(mock_model_manager, sample_data):
    """Test 4: Train all models for a single ticker."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        results = trainer.train_single_ticker(
            ticker='AAPL',
            X=sample_data['X'],
            y_up=sample_data['y_up'],
            y_down=sample_data['y_down']
        )

        # Check that results dictionary is returned
        assert isinstance(results, dict)

        # Verify ModelManager methods were called
        assert mock_model_manager.get_or_create_model.called
        assert mock_model_manager.save_models.called


# ========== Test 5: Batch Ticker Training ==========

def test_batch_ticker_training(mock_model_manager, batch_ticker_data):
    """Test 5: Train models for multiple tickers in batch."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False
        mock_settings.PREDICTION_TARGETS = ['up', 'down']

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        results = trainer.train_ticker_batch(ticker_data=batch_ticker_data)

        # Check results for all tickers
        assert 'AAPL' in results
        assert 'GOOGL' in results
        assert 'MSFT' in results
        assert isinstance(results['AAPL'], dict)


# ========== Test 6: Parallel Worker Configuration ==========

def test_parallel_worker_configuration(mock_model_manager):
    """Test 6: Parallel worker count configuration."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 16
        mock_settings.USE_GPU = False

        # Custom worker count
        trainer1 = GPUParallelTrainer(
            model_manager=mock_model_manager,
            n_parallel=4
        )
        assert trainer1.n_parallel == 4

        # Default from settings
        trainer2 = GPUParallelTrainer(model_manager=mock_model_manager)
        assert trainer2.n_parallel == 16


# ========== Test 7: Validation Data Split ==========

def test_validation_data_split(mock_model_manager, sample_data):
    """Test 7: Train/validation data splitting."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(
            model_manager=mock_model_manager,
            validation_split=0.2
        )

        X = sample_data['X']
        y = sample_data['y_up']

        X_train, X_val, y_train, y_val = trainer._train_val_split(X, y)

        # Check split ratios
        assert len(X_train) == int(len(X) * 0.8)
        assert len(X_val) == len(X) - len(X_train)
        assert len(y_train) == len(X_train)
        assert len(y_val) == len(X_val)

        # Check chronological order (no shuffling)
        assert np.array_equal(X_train, X[:len(X_train)])
        assert np.array_equal(X_val, X[len(X_train):])


def test_validation_split_zero(mock_model_manager, sample_data):
    """Test 7b: No validation split when validation_split=0."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(
            model_manager=mock_model_manager,
            validation_split=0.0
        )

        X = sample_data['X']
        y = sample_data['y_up']

        X_train, X_val, y_train, y_val = trainer._train_val_split(X, y)

        assert np.array_equal(X_train, X)
        assert X_val is None
        assert y_val is None


# ========== Test 8: Tree Models Parallel Training ==========

def test_tree_models_parallel_training(mock_model_manager, sample_data):
    """Test 8: Parallel training of tree-based models."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        X_train = sample_data['X'][:80]
        y_train = sample_data['y_up'][:80]
        X_val = sample_data['X'][80:]
        y_val = sample_data['y_up'][80:]

        results = trainer._train_tree_models_parallel(
            ticker='AAPL',
            target='up',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Check that tree models were trained
        assert 'xgboost_up' in results or 'lightgbm_up' in results
        assert isinstance(results, dict)


# ========== Test 9: Neural Models Sequential Training ==========

def test_neural_models_sequential_training(mock_model_manager, sample_data):
    """Test 9: Sequential training of neural models."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        X_train = sample_data['X'][:80]
        y_train = sample_data['y_up'][:80]
        X_val = sample_data['X'][80:]
        y_val = sample_data['y_up'][80:]

        results = trainer._train_neural_models_sequential(
            ticker='AAPL',
            target='down',
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Check that neural models were attempted
        assert isinstance(results, dict)


# ========== Test 10: GPU Memory Management ==========

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gpu_memory_management(mock_model_manager):
    """Test 10: GPU cache clearing and memory management."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.empty_cache') as mock_empty_cache, \
         patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5080'), \
         patch('torch.cuda.get_device_properties') as mock_props:

        mock_settings.USE_GPU = True
        mock_settings.N_PARALLEL_WORKERS = 2

        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1e9
        mock_props.return_value = mock_device_props

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        # Test manual cache clearing
        trainer.clear_gpu_memory()

        assert mock_empty_cache.called


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gpu_memory_stats(mock_model_manager):
    """Test 10b: GPU memory usage statistics."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.memory_allocated', return_value=2 * 1e9), \
         patch('torch.cuda.memory_reserved', return_value=3 * 1e9), \
         patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5080'), \
         patch('torch.cuda.get_device_properties') as mock_props:

        mock_settings.USE_GPU = True
        mock_settings.N_PARALLEL_WORKERS = 2

        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1e9
        mock_props.return_value = mock_device_props

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        stats = trainer.get_gpu_memory_usage()

        assert stats['available'] is True
        assert 'allocated_gb' in stats
        assert 'total_gb' in stats
        assert stats['total_gb'] == 16.0


# ========== Test 11: Checkpoint Saving ==========

def test_checkpoint_saving(mock_model_manager, sample_data):
    """Test 11: Model checkpoint saving after training."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        trainer.train_single_ticker(
            ticker='AAPL',
            X=sample_data['X'],
            y_up=sample_data['y_up'],
            y_down=sample_data['y_down']
        )

        # Verify save_models was called
        mock_model_manager.save_models.assert_called_with('AAPL')


# ========== Test 12: Training History Tracking ==========

def test_training_history_tracking(mock_model_manager, sample_data):
    """Test 12: Training history recording and retrieval."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        # Record training events manually
        trainer._record_training('AAPL', 'xgboost', 'up', 100)
        trainer._record_training('AAPL', 'lightgbm', 'up', 100)
        trainer._record_training('GOOGL', 'xgboost', 'down', 150)

        # Check training history
        assert 'AAPL_xgboost_up' in trainer.training_history
        assert len(trainer.training_history['AAPL_xgboost_up']) == 1
        assert trainer.training_history['AAPL_xgboost_up'][0]['n_samples'] == 100

        # Get summary
        summary = trainer.get_training_summary()
        assert summary['total_trainings'] == 3
        assert summary['unique_tickers'] == 2
        assert 'AAPL' in summary['tickers']
        assert 'GOOGL' in summary['tickers']


# ========== Test 13: Error Recovery ==========

def test_error_recovery_continue_on_failure(mock_model_manager, sample_data):
    """Test 13: Training continues when a single model fails."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        # Configure mock to fail for one model type
        def get_or_create_side_effect(ticker, model_type, target):
            mock_model = Mock()
            if model_type == 'xgboost':
                # Simulate training failure
                mock_model.train = Mock(side_effect=Exception("Training failed"))
            else:
                mock_model.train = Mock(return_value=None)
            mock_model.is_trained = True
            return (model_type, mock_model)

        mock_model_manager.get_or_create_model.side_effect = get_or_create_side_effect

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        # Training should continue despite XGBoost failure
        results = trainer._train_tree_models_parallel(
            ticker='AAPL',
            target='up',
            X_train=sample_data['X'][:80],
            y_train=sample_data['y_up'][:80]
        )

        # XGBoost should have failed, but LightGBM should succeed
        assert 'xgboost_up' in results
        assert results['xgboost_up'] is False  # Failed


# ========== Test 14: Memory Cleanup ==========

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_memory_cleanup(mock_model_manager, sample_data):
    """Test 14: Garbage collection and memory cleanup."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings, \
         patch('gc.collect') as mock_gc_collect, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.empty_cache') as mock_empty_cache, \
         patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5080'), \
         patch('torch.cuda.get_device_properties') as mock_props:

        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = True

        mock_device_props = Mock()
        mock_device_props.total_memory = 16 * 1e9
        mock_props.return_value = mock_device_props

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        # Train models (should trigger gc.collect when GPU is available)
        trainer._train_neural_models_sequential(
            ticker='AAPL',
            target='up',
            X_train=sample_data['X'][:80],
            y_train=sample_data['y_up'][:80]
        )

        # gc.collect should be called during neural model training with GPU
        assert mock_gc_collect.called
        assert mock_empty_cache.called


# ========== Test 15: Long Training Simulation ==========

def test_long_training_simulation(mock_model_manager):
    """Test 15: Simulate long-running training with multiple tickers."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False
        mock_settings.PREDICTION_TARGETS = ['up', 'down']

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        # Generate data for multiple tickers
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        ticker_data = {}

        for ticker in tickers:
            np.random.seed(hash(ticker) % 2**32)
            X = np.random.randn(200, 57)
            y = np.random.randint(0, 2, 200)
            ticker_data[ticker] = (X, y)

        # Train all tickers
        results = trainer.train_ticker_batch(ticker_data)

        # Verify all tickers were processed
        assert len(results) == len(tickers)
        for ticker in tickers:
            assert ticker in results
            assert isinstance(results[ticker], dict)


# ========== Test 16: Integration Test ==========

def test_integration_full_pipeline(mock_model_manager, sample_data):
    """Test 16: Full integration test of entire training pipeline."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        # Create trainer
        trainer = GPUParallelTrainer(
            model_manager=mock_model_manager,
            n_parallel=2,
            validation_split=0.2
        )

        # Train single ticker with all labels (Structure A + B)
        results = trainer.train_single_ticker(
            ticker='AAPL',
            X=sample_data['X'],
            y_up=sample_data['y_up'],
            y_down=sample_data['y_down'],
            y_volatility=sample_data['y_volatility'],
            y_direction=sample_data['y_direction']
        )

        # Verify results
        assert isinstance(results, dict)

        # Verify ModelManager interactions
        assert mock_model_manager.get_or_create_model.called
        assert mock_model_manager.save_models.called

        # Check training history
        assert len(trainer.training_history) > 0

        # Get training summary
        summary = trainer.get_training_summary()
        assert summary['total_trainings'] > 0
        assert 'AAPL' in summary['tickers']

        # Test memory cleanup
        trainer.clear_gpu_memory()

        # Test GPU stats (should handle CPU-only case)
        gpu_stats = trainer.get_gpu_memory_usage()
        assert 'available' in gpu_stats


def test_integration_batch_pipeline(mock_model_manager):
    """Test 16b: Full batch training integration test."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 4
        mock_settings.USE_GPU = False
        mock_settings.PREDICTION_TARGETS = ['up', 'down']

        trainer = GPUParallelTrainer(
            model_manager=mock_model_manager,
            validation_split=0.15
        )

        # Create batch data
        batch_data = {}
        for ticker in ['AAPL', 'GOOGL', 'MSFT']:
            np.random.seed(hash(ticker) % 2**32)
            X = np.random.randn(150, 57)
            y = np.random.randint(0, 2, 150)
            batch_data[ticker] = (X, y)

        # Train batch
        results = trainer.train_ticker_batch(batch_data)

        # Verify all tickers completed
        assert len(results) == 3
        assert all(ticker in results for ticker in batch_data.keys())

        # Get final summary
        summary = trainer.get_training_summary()
        assert summary['unique_tickers'] > 0
        assert summary['total_trainings'] > 0


# ========== Additional Edge Case Tests ==========

def test_empty_training_history(mock_model_manager):
    """Test training summary with no training events."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        summary = trainer.get_training_summary()
        assert summary['total_trainings'] == 0
        assert summary['unique_tickers'] == 0


def test_hybrid_training_method(mock_model_manager, sample_data):
    """Test hybrid training entry point."""
    with patch('src.trainer.gpu_trainer.settings') as mock_settings:
        mock_settings.N_PARALLEL_WORKERS = 2
        mock_settings.USE_GPU = False

        trainer = GPUParallelTrainer(model_manager=mock_model_manager)

        results = trainer.train_single_ticker_hybrid(
            ticker='AAPL',
            X=sample_data['X'],
            y_up=sample_data['y_up'],
            y_down=sample_data['y_down'],
            y_volatility=sample_data['y_volatility'],
            y_direction=sample_data['y_direction']
        )

        assert isinstance(results, dict)
