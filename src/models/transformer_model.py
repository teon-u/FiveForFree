"""Transformer model for NASDAQ prediction."""

from pathlib import Path
from typing import Optional
import numpy as np
import math
import pickle

from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from config.settings import settings

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. TransformerModel will not work.")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """Transformer for time series classification."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # Note: Sigmoid removed - use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global average pooling
        x = x.mean(dim=1)
        out = self.fc(x)
        return out.squeeze()


# Set the alias for backward compatibility
TransformerNetwork = TransformerClassifier


class TransformerModel(BaseModel):
    """Transformer-based prediction model."""

    def __init__(
        self,
        ticker: str,
        target: str,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = None,  # None이면 settings에서 가져옴
        sequence_length: int = 60,
        early_stopping_patience: int = 5,
        min_delta: float = 0.001,
        **kwargs
    ):
        """
        Initialize Transformer model.

        Args:
            ticker: Ticker symbol
            target: Prediction target (up/down)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            sequence_length: Input sequence length
            early_stopping_patience: Epochs to wait before early stop
            min_delta: Minimum change to qualify as improvement
        """
        super().__init__(ticker=ticker, target=target, model_type='transformer')

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size if batch_size is not None else settings.TRANSFORMER_BATCH_SIZE
        self.sequence_length = sequence_length
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self._model: Optional[TransformerClassifier] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else None
        self.input_size: Optional[int] = None  # Set during training, used for loading
        self._scaler: Optional[StandardScaler] = None  # Feature scaler for normalization
        self._best_val_loss: float = float('inf')  # Track best validation loss

    def _prepare_sequences(self, X, y=None):
        """Prepare sequences for Transformer."""
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
        """Train the Transformer model with early stopping."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")

        # Handle NaN/Inf in raw data
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fit scaler on training data and transform
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Prepare sequences with scaled data
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)

        # Prepare validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
            X_val_scaled = self._scaler.transform(X_val)
            X_val_seq, y_val_seq = self._prepare_sequences(X_val_scaled, y_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val_seq)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        self.input_size = X_seq.shape[2]
        self._model = TransformerClassifier(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        # Use BCEWithLogitsLoss for numerical stability (combines Sigmoid + BCELoss)
        criterion = nn.BCEWithLogitsLoss()

        # Create DataLoader for efficient GPU utilization
        # Windows에서는 num_workers=0 사용 (multiprocessing 호환성)
        use_cuda = self._device.type == 'cuda'
        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows 호환성
            pin_memory=use_cuda,  # GPU 사용 시 CPU→GPU 전송 최적화
            drop_last=True
        )

        # Mixed Precision (AMP) for RTX GPU optimization
        scaler = GradScaler(enabled=use_cuda)

        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        # Training loop with AMP and early stopping
        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # 비동기 전송으로 오버랩
                batch_X = batch_X.to(self._device, non_blocking=True)
                batch_y = batch_y.to(self._device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # 더 효율적인 메모리 초기화

                # Mixed precision forward pass
                with autocast(enabled=use_cuda):
                    outputs = self._model(batch_X)
                    loss = criterion(outputs, batch_y)

                epoch_loss += loss.item()

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            # Validation and early stopping
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self._device)
                        batch_y = batch_y.to(self._device)
                        with autocast(enabled=use_cuda):
                            outputs = self._model(batch_X)
                            val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_loader)
                self._model.train()

                # Check for improvement
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}/{self.epochs} (val_loss: {val_loss:.4f})")
                        break

        # Restore best model if early stopping was used
        if best_model_state is not None:
            self._model.load_state_dict({k: v.to(self._device) for k, v in best_model_state.items()})
            self._best_val_loss = best_val_loss

        self.is_trained = True
        self._update_last_trained()

        # Calculate and store train accuracy using validation set
        try:
            if X_val is not None and y_val is not None:
                y_pred = (self.predict_proba(np.asarray(X_val)) >= 0.5).astype(int)
                y_val_arr = np.asarray(y_val)
                # Handle sequence length mismatch
                min_len = min(len(y_pred), len(y_val_arr))
                self.train_accuracy = float(np.mean(y_pred[:min_len] == y_val_arr[:min_len]))
                logger.info(f"Transformer model trained for {self.ticker} {self.target} (val_acc={self.train_accuracy:.2%})")
            else:
                logger.info(f"Transformer model trained for {self.ticker} {self.target}")
        except Exception as e:
            logger.warning(f"Could not calculate train_accuracy for Transformer {self.ticker} {self.target}: {e}")
            logger.info(f"Transformer model trained for {self.ticker} {self.target}")

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
            self._model = TransformerClassifier(
                input_size=self.input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self._device)
            self._model.load_state_dict(state_dict)

        # Load scaler
        scaler_path = path.with_suffix('.scaler')
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
