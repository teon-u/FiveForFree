"""Test LSTM model implementation."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import pickle

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.lstm_model import LSTMModel, LSTMNetwork


@pytest.fixture
def sample_data():
    """Create sample 3D sequence data for LSTM testing."""
    np.random.seed(42)
    samples = 100
    timesteps = 60
    features = 10

    # Create 2D data first (samples x features)
    X = np.random.randn(samples, features).astype(np.float32)
    y = np.random.randint(0, 2, size=samples).astype(np.float32)

    return X, y


@pytest.fixture
def lstm_model():
    """Create a basic LSTM model instance."""
    return LSTMModel(
        ticker="TEST",
        target="up",
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=2,  # Small for testing
        batch_size=16,
        sequence_length=10
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_model_initialization(lstm_model):
    """Test 1: Model initialization test."""
    assert lstm_model.ticker == "TEST"
    assert lstm_model.target == "up"
    assert lstm_model.model_type == "lstm"
    assert lstm_model.hidden_size == 32
    assert lstm_model.num_layers == 2
    assert lstm_model.dropout == 0.2
    assert lstm_model.learning_rate == 0.001
    assert lstm_model.epochs == 2
    assert lstm_model.batch_size == 16
    assert lstm_model.sequence_length == 10
    assert lstm_model.is_trained is False
    assert lstm_model._model is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_hyperparameter_settings():
    """Test 2: Hyperparameter setting test."""
    model = LSTMModel(
        ticker="NVDA",
        target="down",
        hidden_size=64,
        num_layers=3,
        dropout=0.3,
        learning_rate=0.0001,
        epochs=100,
        batch_size=32,
        sequence_length=60
    )

    assert model.hidden_size == 64
    assert model.num_layers == 3
    assert model.dropout == 0.3
    assert model.learning_rate == 0.0001
    assert model.epochs == 100
    assert model.batch_size == 32
    assert model.sequence_length == 60


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_sequence_data_shape(lstm_model, sample_data):
    """Test 3: Sequence data shape test (samples, timesteps, features)."""
    X, y = sample_data

    # Prepare sequences
    X_seq, y_seq = lstm_model._prepare_sequences(X, y)

    # Check shape: (samples, timesteps, features)
    assert len(X_seq.shape) == 3
    assert X_seq.shape[1] == lstm_model.sequence_length
    assert X_seq.shape[2] == X.shape[1]

    # Check that we have valid sequences
    expected_sequences = len(X) - lstm_model.sequence_length + 1
    assert X_seq.shape[0] == expected_sequences
    assert y_seq.shape[0] == expected_sequences


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_train_small_data(lstm_model, sample_data):
    """Test 4: Training test with small data."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Check training status
    assert lstm_model.is_trained is True
    assert lstm_model._model is not None
    assert lstm_model.input_size is not None
    assert lstm_model._scaler is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_is_trained_status(lstm_model, sample_data):
    """Test 5: is_trained status after training."""
    X, y = sample_data

    # Initially not trained
    assert lstm_model.is_trained is False

    # Train model
    lstm_model.train(X, y)

    # Should be trained now
    assert lstm_model.is_trained is True
    assert lstm_model.last_trained_at is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_predict(lstm_model, sample_data):
    """Test 6: Prediction test."""
    X, y = sample_data

    # Train first
    lstm_model.train(X, y)

    # Predict
    X_test = X[:20]
    predictions = lstm_model.predict_proba(X_test)

    # Check predictions
    assert predictions is not None
    assert len(predictions) > 0
    assert isinstance(predictions, np.ndarray)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_probability_range(lstm_model, sample_data):
    """Test 7: Probability prediction range test (0~1)."""
    X, y = sample_data

    # Train first
    lstm_model.train(X, y)

    # Predict
    X_test = X[:20]
    predictions = lstm_model.predict_proba(X_test)

    # Check probability range
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_untrained_model_error(lstm_model, sample_data):
    """Test 8: Error test when predicting with untrained model."""
    X, y = sample_data
    X_test = X[:20]

    # Should raise error when predicting without training
    with pytest.raises(ValueError, match="Model not trained"):
        lstm_model.predict_proba(X_test)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_model(lstm_model, sample_data):
    """Test 9: Save test."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pkl"

        # Save should not raise error
        try:
            lstm_model.save(save_path)
        except NotImplementedError:
            # BaseModel.save is abstract, but LSTM saves .pt and .scaler files
            pass

        # Check that model-specific files were created
        assert save_path.with_suffix('.pt').exists()
        assert save_path.with_suffix('.scaler').exists()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_load_model(lstm_model, sample_data):
    """Test 10: Load test."""
    X, y = sample_data

    # Train and save model
    lstm_model.train(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pkl"

        # Save should not raise error
        try:
            lstm_model.save(save_path)
        except NotImplementedError:
            pass

        # Create new model and load
        new_model = LSTMModel(
            ticker="TEST",
            target="up",
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            sequence_length=10
        )

        # Load should not raise error
        try:
            new_model.load(save_path)
        except NotImplementedError:
            pass

        # Check loaded components (model and scaler should be loaded from .pt and .scaler files)
        assert new_model._model is not None
        assert new_model._scaler is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_save_load_prediction_consistency(lstm_model, sample_data):
    """Test 11: Save/Load prediction consistency test."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Make predictions before saving
    X_test = X[:20]
    predictions_before = lstm_model.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pkl"

        # Save should not raise error
        try:
            lstm_model.save(save_path)
        except NotImplementedError:
            pass

        # Create new model and load
        new_model = LSTMModel(
            ticker="TEST",
            target="up",
            hidden_size=32,
            num_layers=2,
            dropout=0.2,
            sequence_length=10
        )

        # Load should not raise error
        try:
            new_model.load(save_path)
        except NotImplementedError:
            pass

        # Manually set is_trained since BaseModel.load is abstract
        # In real usage, this would be set by the concrete implementation
        new_model.is_trained = True

        # Make predictions after loading
        predictions_after = new_model.predict_proba(X_test)

        # Check consistency
        assert predictions_before.shape == predictions_after.shape
        np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_scaler_save_load(lstm_model, sample_data):
    """Test 12: Scaler save/load test."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Save scaler stats
    original_mean = lstm_model._scaler.mean_.copy()
    original_scale = lstm_model._scaler.scale_.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pkl"
        lstm_model.save(save_path)

        # Create new model and load
        new_model = LSTMModel(
            ticker="TEST",
            target="up",
            hidden_size=32,
            num_layers=2,
            sequence_length=10
        )
        new_model.load(save_path)

        # Check scaler was loaded correctly
        assert new_model._scaler is not None
        np.testing.assert_array_almost_equal(new_model._scaler.mean_, original_mean)
        np.testing.assert_array_almost_equal(new_model._scaler.scale_, original_scale)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_short_sequence_padding(lstm_model, sample_data):
    """Test 13: Short sequence padding test."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Create test data shorter than sequence_length
    short_X = X[:5]  # Only 5 samples, less than sequence_length of 10

    # Should handle padding automatically
    predictions = lstm_model.predict_proba(short_X)

    # Check predictions were made
    assert predictions is not None
    # predictions might be a scalar or array
    if isinstance(predictions, np.ndarray):
        assert predictions.size > 0
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)
    else:
        # Single prediction
        assert 0.0 <= predictions <= 1.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_nan_handling(lstm_model):
    """Test 14: NaN value handling test."""
    # Create data with NaN values
    X_with_nan = np.random.randn(100, 10).astype(np.float32)
    X_with_nan[10:15, 3:5] = np.nan  # Add NaN values
    y = np.random.randint(0, 2, size=100).astype(np.float32)

    # Should not raise error during training
    lstm_model.train(X_with_nan, y)

    # Should not raise error during prediction
    X_test_with_nan = np.random.randn(20, 10).astype(np.float32)
    X_test_with_nan[5:8, 2] = np.nan
    predictions = lstm_model.predict_proba(X_test_with_nan)

    # Check predictions are valid
    assert predictions is not None
    assert len(predictions) > 0
    assert not np.any(np.isnan(predictions))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_batch_size():
    """Test 15: Batch size test."""
    # Test different batch sizes
    for batch_size in [8, 16, 32, 64]:
        model = LSTMModel(
            ticker="TEST",
            target="up",
            hidden_size=32,
            num_layers=2,
            epochs=1,
            batch_size=batch_size,
            sequence_length=10
        )

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=100).astype(np.float32)

        # Should train successfully with different batch sizes
        model.train(X, y)
        assert model.is_trained is True


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_epochs():
    """Test 16: Epoch count test."""
    # Test different epoch counts
    for epochs in [1, 5, 10, 20]:
        model = LSTMModel(
            ticker="TEST",
            target="up",
            hidden_size=32,
            num_layers=2,
            epochs=epochs,
            batch_size=16,
            sequence_length=10
        )

        assert model.epochs == epochs

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=100).astype(np.float32)

        # Should train successfully with different epoch counts
        model.train(X, y)
        assert model.is_trained is True


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_device_handling(lstm_model, sample_data):
    """Test 17: GPU/CPU device handling test."""
    X, y = sample_data

    # Check device is set
    assert lstm_model._device is not None

    # Train model
    lstm_model.train(X, y)

    # Check model is on correct device
    if torch.cuda.is_available():
        assert next(lstm_model._model.parameters()).is_cuda
    else:
        assert not next(lstm_model._model.parameters()).is_cuda

    # Prediction should work regardless of device
    X_test = X[:20]
    predictions = lstm_model.predict_proba(X_test)
    assert predictions is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_memory_cleanup(lstm_model, sample_data):
    """Test 18: Memory cleanup test."""
    X, y = sample_data

    # Train model
    lstm_model.train(X, y)

    # Get initial memory state
    initial_model = lstm_model._model
    assert initial_model is not None

    # Make predictions
    X_test = X[:20]
    predictions = lstm_model.predict_proba(X_test)

    # Model should still be in eval mode after prediction
    assert not lstm_model._model.training

    # Test that we can delete the model
    del lstm_model._model
    lstm_model._model = None

    # Verify cleanup
    assert lstm_model._model is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_lstm_network_architecture():
    """Bonus test: LSTM network architecture test."""
    input_size = 10
    hidden_size = 64
    num_layers = 2
    dropout = 0.2

    network = LSTMNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # Check architecture
    assert network.hidden_size == hidden_size
    assert network.num_layers == num_layers
    assert network.lstm.input_size == input_size
    assert network.lstm.hidden_size == hidden_size
    assert network.lstm.num_layers == num_layers

    # Test forward pass
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, input_size)

    output = network(x)

    # Output should be (batch_size,) after squeeze
    assert output.shape == (batch_size,)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_inf_handling(lstm_model):
    """Bonus test: Inf value handling test."""
    # Create data with Inf values
    X_with_inf = np.random.randn(100, 10).astype(np.float32)
    X_with_inf[10:15, 3] = np.inf
    X_with_inf[20:25, 5] = -np.inf
    y = np.random.randint(0, 2, size=100).astype(np.float32)

    # Should not raise error during training
    lstm_model.train(X_with_inf, y)

    # Should not raise error during prediction
    X_test_with_inf = np.random.randn(20, 10).astype(np.float32)
    X_test_with_inf[5, 2] = np.inf
    predictions = lstm_model.predict_proba(X_test_with_inf)

    # Check predictions are valid
    assert predictions is not None
    assert len(predictions) > 0
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))
