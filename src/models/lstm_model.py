"""LSTM model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np
import pickle

from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. LSTMModel will not work.")


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series classification."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # Note: Sigmoid removed - use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


class LSTMModel(BaseModel):
    """LSTM-based prediction model."""

    def __init__(
        self,
        ticker: str,
        target: str,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        sequence_length: int = 60,
        **kwargs
    ):
        """
        Initialize LSTM model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            sequence_length: Input sequence length
        """
        super().__init__(ticker=ticker, target=target, model_type='lstm')

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self._model: Optional[LSTMNetwork] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else None
        self.input_size: Optional[int] = None  # Set during training, used for loading
        self._scaler: Optional[StandardScaler] = None  # Feature scaler for normalization

    def _prepare_sequences(self, X, y=None):
        """Prepare sequences for LSTM."""
        sequences = []
        targets = []

        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
            if y is not None:
                targets.append(y[i + self.sequence_length - 1])

        X_seq = np.array(sequences)

        if y is not None:
            y_seq = np.array(targets)
            return X_seq, y_seq
        return X_seq

    def train(self, X, y, X_val=None, y_val=None):
        """Train the LSTM model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")

        # Handle NaN/Inf in raw data
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fit scaler on training data and transform
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Prepare sequences with scaled data
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)

        # Initialize model
        self.input_size = X_seq.shape[2]
        self._model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        # Use BCEWithLogitsLoss for numerical stability (combines Sigmoid + BCELoss)
        criterion = nn.BCEWithLogitsLoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self._device)
        y_tensor = torch.FloatTensor(y_seq).to(self._device)

        # Training loop
        self._model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i + self.batch_size]
                batch_y = y_tensor[i:i + self.batch_size]

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

        self.is_trained = True
        self._update_last_trained()
        logger.info(f"LSTM model trained for {self.ticker} {self.target}")

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self._model is None:
            raise ValueError("Model not trained")
        if self._scaler is None:
            raise ValueError("Scaler not initialized")

        self._model.eval()

        # Handle NaN/Inf and scale
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self._scaler.transform(X)

        # Handle single sample or batch
        if len(X_scaled) < self.sequence_length:
            # Pad by repeating first value instead of zeros (zeros cause meaningless predictions)
            first_value = X_scaled[0:1]  # Shape: (1, n_features)
            needed = self.sequence_length - len(X_scaled)
            padding = np.repeat(first_value, needed, axis=0)  # Shape: (needed, n_features)
            padded = np.vstack([padding, X_scaled])  # Shape: (sequence_length, n_features)
            X_seq = padded.reshape(1, self.sequence_length, -1)
        else:
            X_seq = self._prepare_sequences(X_scaled)

        X_tensor = torch.FloatTensor(X_seq).to(self._device)

        with torch.no_grad():
            logits = self._model(X_tensor)
            # Apply sigmoid since model outputs logits
            proba = torch.sigmoid(logits).cpu().numpy()

        return proba

    def save(self, path: Path):
        """Save model to disk."""
        if self._model is not None and HAS_TORCH:
            # Save model state with input_size for proper loading
            save_dict = {
                'state_dict': self._model.state_dict(),
                'input_size': self.input_size
            }
            torch.save(save_dict, path.with_suffix('.pt'))
        # Save scaler separately
        if self._scaler is not None:
            with open(path.with_suffix('.scaler'), 'wb') as f:
                pickle.dump(self._scaler, f)
        super().save(path)

    def load(self, path: Path):
        """Load model from disk."""
        super().load(path)
        pt_path = path.with_suffix('.pt')
        if pt_path.exists() and HAS_TORCH:
            checkpoint = torch.load(pt_path, map_location=self._device)

            # Handle both old format (just state_dict) and new format (dict with input_size)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.input_size = checkpoint.get('input_size', 49)
            else:
                # Old format - just state_dict
                state_dict = checkpoint
                self.input_size = self.input_size or 49  # Use existing or default

            # Initialize model with correct input_size
            self._model = LSTMNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self._device)
            self._model.load_state_dict(state_dict)

        # Load scaler
        scaler_path = path.with_suffix('.scaler')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
