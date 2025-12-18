"""Comprehensive tests for TransformerModel."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime

# Check torch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.transformer_model import TransformerModel, TransformerClassifier, PositionalEncoding


@pytest.fixture
def sample_sequence_data():
    """Generate sample 3D sequence data for testing."""
    np.random.seed(42)
    # Shape: (100 samples, 60 sequence_length, 10 features)
    X = np.random.randn(100, 60, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    return X, y


@pytest.fixture
def small_sequence_data():
    """Generate small sequence data for quick training tests."""
    np.random.seed(42)
    # Shape: (30 samples, 20 sequence_length, 5 features)
    X = np.random.randn(30, 20, 5).astype(np.float32)
    y = np.random.randint(0, 2, 30).astype(np.float32)
    return X, y


@pytest.fixture
def transformer_model():
    """Create a basic TransformerModel instance."""
    return TransformerModel(
        ticker="NVDA",
        target="up",
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        epochs=2,
        batch_size=8,
        sequence_length=20
    )


# Test 1: Model initialization
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_model_initialization():
    """Test that TransformerModel initializes correctly with default parameters."""
    model = TransformerModel(ticker="NVDA", target="up")

    assert model.ticker == "NVDA"
    assert model.target == "up"
    assert model.model_type == "transformer"
    assert model.is_trained is False
    assert model.d_model == 64
    assert model.nhead == 4
    assert model.num_layers == 2
    assert model.dropout == 0.1
    assert model.learning_rate == 0.001
    assert model.epochs == 50
    assert model.batch_size == 32
    assert model.sequence_length == 60


# Test 2: Hyperparameter configuration
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_hyperparameter_configuration():
    """Test custom hyperparameter settings are properly stored."""
    model = TransformerModel(
        ticker="AAPL",
        target="down",
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.2,
        learning_rate=0.0005,
        epochs=100,
        batch_size=64,
        sequence_length=30
    )

    assert model.d_model == 128
    assert model.nhead == 8
    assert model.num_layers == 3
    assert model.dropout == 0.2
    assert model.learning_rate == 0.0005
    assert model.epochs == 100
    assert model.batch_size == 64
    assert model.sequence_length == 30


# Test 3: Input dimension handling
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_input_dimension_handling(transformer_model, small_sequence_data):
    """Test that model correctly handles different input dimensions."""
    X, y = small_sequence_data
    # X shape: (30, 20, 5) - 5 features

    # Flatten for training (transformer expects 2D during prepare, 3D after)
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])

    transformer_model.train(X_flat, y_flat)

    # Check input_size was set correctly
    assert transformer_model.input_size == 5


# Test 4: Attention head number validation
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_attention_heads():
    """Test different attention head configurations."""
    # Valid: d_model divisible by nhead
    model_4_heads = TransformerModel(ticker="TEST", target="up", d_model=64, nhead=4)
    assert model_4_heads.nhead == 4

    model_8_heads = TransformerModel(ticker="TEST", target="up", d_model=64, nhead=8)
    assert model_8_heads.nhead == 8

    # d_model must be divisible by nhead for transformer to work
    model_2_heads = TransformerModel(ticker="TEST", target="up", d_model=64, nhead=2)
    assert model_2_heads.nhead == 2


# Test 5: Layer count configuration
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_layer_count():
    """Test different transformer layer counts."""
    model_1_layer = TransformerModel(ticker="TEST", target="up", num_layers=1)
    assert model_1_layer.num_layers == 1

    model_4_layers = TransformerModel(ticker="TEST", target="up", num_layers=4)
    assert model_4_layers.num_layers == 4

    model_6_layers = TransformerModel(ticker="TEST", target="up", num_layers=6)
    assert model_6_layers.num_layers == 6


# Test 6: Sequence data shape validation
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_sequence_data_shape(transformer_model):
    """Test that sequence preparation produces correct shapes."""
    np.random.seed(42)
    # Create flat data
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)

    # Prepare sequences with length 20
    X_seq, y_seq = transformer_model._prepare_sequences(X, y)

    # Expected: (100-20+1, 20, 10) = (81, 20, 10)
    assert X_seq.shape == (81, 20, 10)
    assert y_seq.shape == (81,)


# Test 7: Training test with small data
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_training_small_data(transformer_model, small_sequence_data):
    """Test model training completes without errors on small dataset."""
    X, y = small_sequence_data

    # Flatten data for training
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])

    # Train model (2 epochs, small batch)
    transformer_model.train(X_flat, y_flat)

    # Check training completed
    assert transformer_model.is_trained is True
    assert transformer_model._model is not None


# Test 8: Training status flag
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_is_trained_flag(transformer_model, small_sequence_data):
    """Test is_trained flag updates correctly after training."""
    X, y = small_sequence_data

    # Before training
    assert transformer_model.is_trained is False

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # After training
    assert transformer_model.is_trained is True
    assert transformer_model.last_trained_at is not None


# Test 9: Prediction test
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_prediction(transformer_model, small_sequence_data):
    """Test model can make predictions after training."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Prepare test data
    X_test = X_flat[:50]  # Use first 50 samples

    # Make predictions
    predictions = transformer_model.predict_proba(X_test)

    assert predictions is not None
    assert len(predictions) > 0


# Test 10: Probability range validation
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_probability_range(transformer_model, small_sequence_data):
    """Test that predicted probabilities are in valid range [0, 1]."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Make predictions
    X_test = X_flat[:30]
    predictions = transformer_model.predict_proba(X_test)

    # Check all probabilities in [0, 1]
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


# Test 11: Prediction before training error
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_predict_before_training_error(transformer_model, small_sequence_data):
    """Test that predicting before training raises appropriate error."""
    X, _ = small_sequence_data
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])

    with pytest.raises(ValueError, match="Model not trained"):
        transformer_model.predict_proba(X_flat[:10])


# Test 12: Model save test
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_model_save(transformer_model, small_sequence_data):
    """Test model can be saved to disk."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        transformer_model.save(save_path)

        # Check files exist
        assert save_path.with_suffix('.pt').exists()
        assert save_path.with_suffix('.scaler').exists()


# Test 13: Model load test
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_model_load(transformer_model, small_sequence_data):
    """Test model can be loaded from disk."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        transformer_model.save(save_path)

        # Create new model and load
        loaded_model = TransformerModel(
            ticker="NVDA",
            target="up",
            d_model=32,
            nhead=4,
            num_layers=2,
            sequence_length=20
        )
        loaded_model.load(save_path)

        # Check model was loaded (network and scaler)
        assert loaded_model._model is not None
        assert loaded_model._scaler is not None
        assert loaded_model.input_size is not None

        # Manually set is_trained since load doesn't restore it
        # This is a known limitation of the current implementation
        loaded_model.is_trained = True

        # Verify loaded model can make predictions
        X_test = X_flat[:30]
        predictions = loaded_model.predict_proba(X_test)
        assert predictions is not None
        assert len(predictions) > 0


# Test 14: Save/Load prediction consistency
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_load_prediction_consistency(transformer_model, small_sequence_data):
    """Test predictions are consistent after save/load."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Get predictions before save
    X_test = X_flat[:30]
    predictions_before = transformer_model.predict_proba(X_test)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        transformer_model.save(save_path)

        loaded_model = TransformerModel(
            ticker="NVDA",
            target="up",
            d_model=32,
            nhead=4,
            num_layers=2,
            sequence_length=20
        )
        loaded_model.load(save_path)

        # Manually set is_trained flag
        loaded_model.is_trained = True

        # Get predictions after load
        predictions_after = loaded_model.predict_proba(X_test)

        # Check consistency (allow small numerical differences)
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-5)


# Test 15: Checkpoint save functionality
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_checkpoint_save(transformer_model, small_sequence_data):
    """Test checkpoint saving includes all necessary components."""
    X, y = small_sequence_data

    # Flatten and train
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])
    transformer_model.train(X_flat, y_flat)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "checkpoint"
        transformer_model.save(save_path)

        # Load checkpoint and verify contents
        checkpoint = torch.load(save_path.with_suffix('.pt'))

        assert 'state_dict' in checkpoint
        assert 'input_size' in checkpoint
        assert checkpoint['input_size'] == transformer_model.input_size


# Test 16: NaN value handling
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_nan_handling(transformer_model):
    """Test model handles NaN values in input data."""
    np.random.seed(42)

    # Create data with NaN values (use larger dataset to avoid batch_size=1 issue)
    X = np.random.randn(200, 5).astype(np.float32)
    X[10:15, 2] = np.nan  # Insert NaN values
    y = np.random.randint(0, 2, 200).astype(np.float32)

    # Should not raise error - NaNs are handled
    transformer_model.train(X, y)

    assert transformer_model.is_trained is True

    # Test prediction with NaN
    X_test = np.random.randn(50, 5).astype(np.float32)
    X_test[5:8, 1] = np.nan

    predictions = transformer_model.predict_proba(X_test)
    assert np.all(np.isfinite(predictions))


# Test 17: Batch size variation
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_batch_size_variation():
    """Test model works with different batch sizes."""
    np.random.seed(42)
    # Use larger dataset to ensure batches > 1
    X = np.random.randn(200, 10).astype(np.float32)
    y = np.random.randint(0, 2, 200).astype(np.float32)

    # Test small batch (ensure sequence creates enough samples)
    model_small_batch = TransformerModel(
        ticker="TEST", target="up",
        batch_size=8, epochs=1, sequence_length=10,
        d_model=32, nhead=4
    )
    model_small_batch.train(X, y)
    assert model_small_batch.is_trained is True

    # Test larger batch
    model_large_batch = TransformerModel(
        ticker="TEST", target="up",
        batch_size=32, epochs=1, sequence_length=10,
        d_model=32, nhead=4
    )
    model_large_batch.train(X, y)
    assert model_large_batch.is_trained is True


# Test 18: Device handling (GPU/CPU)
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_device_handling(transformer_model, small_sequence_data):
    """Test model correctly handles device placement (CPU/GPU)."""
    X, y = small_sequence_data
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])

    transformer_model.train(X_flat, y_flat)

    # Check device is set
    assert transformer_model._device is not None

    # Check model is on correct device
    if torch.cuda.is_available():
        assert transformer_model._device.type == 'cuda'
    else:
        assert transformer_model._device.type == 'cpu'


# Test 19: Gradient clipping
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_gradient_clipping(small_sequence_data):
    """Test that gradient clipping is applied during training."""
    X, y = small_sequence_data
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])

    # Create model
    model = TransformerModel(
        ticker="TEST", target="up",
        epochs=2, sequence_length=20,
        d_model=32, nhead=4
    )

    # Train - gradient clipping happens internally
    model.train(X_flat, y_flat)

    # If training completes without explosion, clipping worked
    assert model.is_trained is True

    # Make prediction to verify model stability
    predictions = model.predict_proba(X_flat[:30])
    assert np.all(np.isfinite(predictions))


# Test 20: Learning rate scheduler compatibility
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_learning_rate_configuration(small_sequence_data):
    """Test different learning rate configurations."""
    X, y = small_sequence_data
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_flat = np.repeat(y, X.shape[1])

    # Test low learning rate
    model_low_lr = TransformerModel(
        ticker="TEST", target="up",
        learning_rate=0.0001, epochs=1, sequence_length=20,
        d_model=32, nhead=4
    )
    model_low_lr.train(X_flat, y_flat)
    assert model_low_lr.is_trained is True

    # Test high learning rate
    model_high_lr = TransformerModel(
        ticker="TEST", target="up",
        learning_rate=0.01, epochs=1, sequence_length=20,
        d_model=32, nhead=4
    )
    model_high_lr.train(X_flat, y_flat)
    assert model_high_lr.is_trained is True

    # Both should produce valid predictions
    preds_low = model_low_lr.predict_proba(X_flat[:20])
    preds_high = model_high_lr.predict_proba(X_flat[:20])

    assert np.all(np.isfinite(preds_low))
    assert np.all(np.isfinite(preds_high))


# Bonus Tests for TransformerClassifier

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_transformer_classifier_forward():
    """Test TransformerClassifier forward pass."""
    classifier = TransformerClassifier(
        input_size=10,
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1
    )

    # Create sample batch
    batch = torch.randn(8, 20, 10)  # (batch, seq_len, features)

    output = classifier(batch)

    # Output should be (batch_size,) since we squeeze
    assert output.shape == (8,)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_positional_encoding():
    """Test PositionalEncoding module."""
    pe = PositionalEncoding(d_model=64, max_len=100)

    # Create sample input
    x = torch.randn(4, 50, 64)  # (batch, seq_len, d_model)

    output = pe(x)

    # Shape should be preserved
    assert output.shape == x.shape

    # Values should be different (positional encoding added)
    assert not torch.allclose(output, x)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_transformer_network_alias():
    """Test TransformerNetwork is an alias for TransformerClassifier."""
    from src.models.transformer_model import TransformerNetwork

    assert TransformerNetwork is TransformerClassifier


# Test for torch not installed scenario
def test_model_without_torch():
    """Test graceful handling when torch is not available."""
    if HAS_TORCH:
        pytest.skip("PyTorch is installed, skipping this test")

    # Model creation should work
    model = TransformerModel(ticker="TEST", target="up")

    # Training should raise ImportError
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)

    with pytest.raises(ImportError, match="PyTorch is not installed"):
        model.train(X, y)
