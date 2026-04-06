from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class _LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence reconstruction.

    The encoder compresses input sequences into a fixed-size latent
    representation, and the decoder reconstructs the original sequence.
    High reconstruction error indicates anomalous patterns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode: compress to latent representation
        _, (hidden, cell) = self.encoder(x)

        # decode: reconstruct from latent
        seq_len = x.size(1)
        # repeat the latent vector across time steps
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        reconstruction = self.output_layer(decoder_output)
        return reconstruction


class LSTMAutoencoderDetector(BaseDetector):
    """Anomaly detection using LSTM autoencoder reconstruction error.

    Trains an autoencoder to reconstruct normal time-series windows.
    During detection, windows with high reconstruction error (MSE)
    are flagged as anomalous. Good at catching complex temporal
    patterns that statistical methods miss.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 1,
        sequence_length: int = 30,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        threshold_sigma: float = 2.0,
        patience: int = 5,
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            sequence_length=sequence_length,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            threshold_sigma=threshold_sigma,
            patience=patience,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.threshold_sigma = threshold_sigma
        self.patience = patience
        self._model = None
        self._threshold = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences from 1D array."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences)

    def fit(self, data: np.ndarray) -> LSTMAutoencoderDetector:
        data = np.asarray(data, dtype=np.float32)

        # normalize
        self._train_mean = data.mean()
        self._train_std = data.std()
        if self._train_std == 0:
            self._train_std = 1.0
        normalized = (data - self._train_mean) / self._train_std

        sequences = self._create_sequences(normalized)
        # reshape to (n_samples, seq_len, 1) for LSTM
        sequences = sequences.reshape(-1, self.sequence_length, 1)

        tensor_data = torch.FloatTensor(sequences).to(self._device)
        dataset = TensorDataset(tensor_data, tensor_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _LSTMAutoencoder(
            input_dim=1, hidden_dim=self.hidden_dim, n_layers=self.n_layers
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                reconstruction = self._model(batch_x)
                loss = criterion(reconstruction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}, loss={avg_loss:.6f}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, loss={avg_loss:.6f}")

        # compute threshold from training reconstruction errors
        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor_data)
            errors = torch.mean((tensor_data - recon) ** 2, dim=(1, 2)).cpu().numpy()

        self._threshold = errors.mean() + self.threshold_sigma * errors.std()
        self._fitted = True
        logger.info(
            f"LSTM Autoencoder fit: threshold={self._threshold:.6f}, "
            f"mean_error={errors.mean():.6f}"
        )
        return self

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = np.asarray(data, dtype=np.float32)
        if not self._fitted:
            self.fit(data)

        normalized = (data - self._train_mean) / self._train_std
        sequences = self._create_sequences(normalized)
        sequences = sequences.reshape(-1, self.sequence_length, 1)
        tensor_data = torch.FloatTensor(sequences).to(self._device)

        self._model.eval()
        with torch.no_grad():
            recon = self._model(tensor_data)
            seq_errors = torch.mean((tensor_data - recon) ** 2, dim=(1, 2)).cpu().numpy()

        # map sequence-level errors back to per-point scores
        # each point appears in multiple windows, take the max error
        n_points = len(data)
        point_scores = np.zeros(n_points)
        point_counts = np.zeros(n_points)

        for i, error in enumerate(seq_errors):
            for j in range(self.sequence_length):
                idx = i + j
                if idx < n_points:
                    point_scores[idx] = max(point_scores[idx], error)
                    point_counts[idx] += 1

        # normalize to [0, 1]
        max_score = point_scores.max() if point_scores.max() > 0 else 1.0
        scores = point_scores / max_score
        labels = (point_scores > self._threshold).astype(int)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=float(self._threshold),
            metadata={
                "detector": "lstm_autoencoder",
                "hidden_dim": self.hidden_dim,
                "sequence_length": self.sequence_length,
                "anomaly_count": int(labels.sum()),
                "mean_reconstruction_error": float(seq_errors.mean()),
            },
        )
