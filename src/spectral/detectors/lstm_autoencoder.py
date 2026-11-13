from __future__ import annotations

import copy
import logging
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral.detectors.base import BaseDetector, DetectionResult, validate_series

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

    Calibration and scoring semantics:
        The last ``calibration_fraction`` of the training windows (in
        chronological order) are held out of optimisation. Early stopping
        monitors their loss, and the labelling threshold is estimated from
        their *window* reconstruction errors. The threshold is
        ``median + threshold_sigma * (IQR / 1.349)``, a robust equivalent of
        ``mean + k * std`` (for Gaussian errors ``IQR / 1.349`` estimates the
        standard deviation). The IQR-based scale is not inflated by a
        minority of anomalies in the calibration sample, and unlike the MAD
        it still reflects the upper error mode produced by phases the model
        reconstructs less well. ``threshold_sigma`` therefore keeps its
        meaning as a number of standard deviations above the typical error,
        but the location is the median and the scale is robust.

        A point is labelled anomalous only when *every* window that contains
        it exceeds that window threshold: its score is the minimum window
        error among its covering windows. A single spike raises the error of
        all ``sequence_length`` windows that contain it, so the previous
        max-pooled projection turned one spike into a block of false
        positives (the upstream ``max`` projection on the same benchmark
        flagged 90% of a series with 3% spikes). Taking the minimum asks
        whether *any* context explains the point: a normal neighbour of a
        spike keeps a window that excludes it, while the spike itself has no
        such window. Sustained events (level shifts, drifts, noise bursts,
        flatlines) have no clean covering window and remain flagged.

        ``seed`` makes training and calibration reproducible. The held-out
        split is chronological because overlapping windows would leak
        calibration windows into training under a random split.
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
        seed: int = 42,
        calibration_fraction: float = 0.2,
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
            seed=seed,
            calibration_fraction=calibration_fraction,
            **kwargs,
        )
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in (0, 1)")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.threshold_sigma = threshold_sigma
        self.patience = patience
        self.seed = seed
        self.calibration_fraction = calibration_fraction
        self._model = None
        self._threshold = None
        self._calibration_source = None
        self._calibration_sequences = 0
        self._calibration_stats: dict = {}
        self._training_history: list[dict] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sliding window sequences from 1D array."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i : i + self.sequence_length])
        return np.array(sequences)

    def _pool_errors_to_points(
        self, sequence_errors: np.ndarray, n_points: int
    ) -> np.ndarray:
        """Project window errors onto the points they cover, keeping the min.

        The minimum is the error of the best context window available for a
        point. It localises an event instead of smearing it: a spike raises
        every window that contains it, but a neighbour still has a window that
        excludes it. Windows covering a point number at most
        ``sequence_length`` (fewer at the series edges), and every point of a
        series at least ``sequence_length`` long is covered by at least one.
        """
        point_errors = np.full(n_points, np.inf, dtype=np.float64)
        for i, error in enumerate(sequence_errors):
            end = min(i + self.sequence_length, n_points)
            np.minimum(point_errors[i:end], error, out=point_errors[i:end])
        return point_errors

    def _sequence_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Per-window MSE under the current model, in evaluation mode."""
        tensor_data = torch.FloatTensor(sequences).to(self._device)
        chunks = []
        self._model.eval()
        with torch.no_grad():
            for start in range(0, len(tensor_data), self.batch_size):
                batch = tensor_data[start : start + self.batch_size]
                reconstruction = self._model(batch)
                mse = torch.mean((batch - reconstruction) ** 2, dim=(1, 2))
                chunks.append(mse.cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

    @staticmethod
    def _calibrate_threshold(
        errors: np.ndarray, sigma_multiplier: float
    ) -> tuple[float, dict]:
        """Return a robust per-window threshold and the statistics behind it.

        The threshold is ``median + k * (IQR / 1.349)``. For Gaussian data
        ``IQR / 1.349`` is the standard deviation, so this is the robust
        analogue of ``mean + k * std``. The mean and std are unreliable here:
        reconstruction errors are right-skewed, a calibration sample may
        contain a few anomalies, and a phase the model reconstructs less well
        creates an upper mode. The IQR keeps that upper mode in the scale
        estimate (the MAD would centre on the lower mode only) while resisting
        a small number of extreme outliers.
        """
        median = float(np.median(errors))
        q1, q3 = np.percentile(errors, [25, 75])
        iqr = float(q3 - q1)
        robust_sigma = iqr / 1.349 if iqr > 0 else float(np.std(errors))
        if robust_sigma <= 0:
            robust_sigma = 1e-12
        stats = {
            "median_window_error": median,
            "iqr_window_error": iqr,
            "robust_sigma": robust_sigma,
            "n_calibration_windows": int(len(errors)),
            "method": "median + k * (IQR / 1.349)",
        }
        return median + sigma_multiplier * robust_sigma, stats

    def fit(
        self,
        data: np.ndarray,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> LSTMAutoencoderDetector:
        """Train on ``data`` and calibrate the threshold on a chronological holdout.

        ``progress_callback(epoch, epochs, train_loss, monitor_loss)`` is invoked
        once per completed epoch and is used by the workbench to render a live
        progress bar. It is purely observational and cannot affect training.
        """
        data = validate_series(
            data, min_points=self.sequence_length + 2
        ).astype(np.float32)
        self._training_history = []

        torch.manual_seed(self.seed)

        # Chronological split in window space. Overlapping windows mean a
        # random split would leak calibration windows into training.
        n_sequences = len(data) - self.sequence_length + 1
        n_calibration = int(round(n_sequences * self.calibration_fraction))
        n_calibration = min(max(n_calibration, 1), n_sequences - 1)
        n_train = n_sequences - n_calibration

        # Normalisation statistics come from the training portion only.
        train_end = n_train + self.sequence_length - 1
        train_values = data[:train_end]
        self._train_mean = float(train_values.mean())
        std = float(train_values.std())
        self._train_std = std if std > 0 else 1.0

        normalized = (data - self._train_mean) / self._train_std
        sequences = self._create_sequences(normalized).reshape(
            -1, self.sequence_length, 1
        )
        train_sequences = sequences[:n_train]
        calibration_sequences = sequences[n_train:]

        tensor_train = torch.FloatTensor(train_sequences).to(self._device)
        dataset = TensorDataset(tensor_train, tensor_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _LSTMAutoencoder(
            input_dim=1, hidden_dim=self.hidden_dim, n_layers=self.n_layers
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        val_tensor = (
            torch.FloatTensor(calibration_sequences).to(self._device)
            if len(calibration_sequences)
            else None
        )
        val_loader = (
            DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=self.batch_size)
            if val_tensor is not None
            else None
        )

        best_state = copy.deepcopy(self._model.state_dict())
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

            # Monitor held-out reconstruction loss when a calibration split
            # exists, so early stopping does not reward memorisation.
            if val_loader is not None:
                self._model.eval()
                squared_error = 0.0
                n_values = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        diff = (self._model(batch_x) - batch_y) ** 2
                        squared_error += float(diff.sum())
                        n_values += diff.numel()
                monitor_loss = squared_error / n_values
            else:
                monitor_loss = avg_loss

            self._training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(avg_loss),
                    "validation_loss": float(monitor_loss),
                }
            )
            if progress_callback is not None:
                progress_callback(epoch + 1, self.epochs, avg_loss, monitor_loss)

            if monitor_loss < best_loss - 1e-12:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self._model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}, "
                        f"val_loss={monitor_loss:.6f}"
                    )
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, loss={avg_loss:.6f}")

        self._model.load_state_dict(best_state)
        self._model.eval()

        # Calibrate on window errors from data the model did not train on.
        # Point labels compare each point's best covering window with this
        # threshold, so the threshold itself must describe a single window.
        calibration_source = (
            calibration_sequences if len(calibration_sequences) else train_sequences
        )
        window_errors = self._sequence_errors(calibration_source)
        self._threshold, self._calibration_stats = self._calibrate_threshold(
            window_errors, self.threshold_sigma
        )
        self._calibration_source = (
            "holdout" if len(calibration_sequences) else "training"
        )
        self._calibration_sequences = int(len(calibration_source))
        self._fitted = True
        logger.info(
            f"LSTM Autoencoder fit: threshold={self._threshold:.6f} "
            f"({self._calibration_source} calibration, "
            f"{self._calibration_sequences} windows), "
            f"median_window_error={np.median(window_errors):.6f}"
        )
        return self

    def reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """Per-point reconstruction error under an already-fitted model.

        The value for a point is the minimum window MSE among the windows that
        cover it -- the best context the model can find for that point. Returns
        raw MSE in normalised units, the same units as ``threshold``. This
        method never fits; call ``fit`` first.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("detector must be fitted before computing errors")
        data = validate_series(
            data, min_points=self.sequence_length
        ).astype(np.float32)

        normalized = (data - self._train_mean) / self._train_std
        sequences = self._create_sequences(normalized).reshape(
            -1, self.sequence_length, 1
        )
        window_errors = self._sequence_errors(sequences)
        return self._pool_errors_to_points(window_errors, len(data))

    def detect(self, data: np.ndarray) -> DetectionResult:
        data = validate_series(
            data, min_points=self.sequence_length
        ).astype(np.float32)
        if not self._fitted:
            self.fit(data)

        point_errors = self.reconstruction_errors(data)
        max_error = point_errors.max()
        scores = (
            point_errors / max_error
            if max_error > 0
            else np.zeros_like(point_errors)
        )
        labels = (point_errors > self._threshold).astype(int)

        return DetectionResult(
            scores=scores,
            labels=labels,
            threshold=float(self._threshold),
            metadata={
                "detector": "lstm_autoencoder",
                "hidden_dim": self.hidden_dim,
                "sequence_length": self.sequence_length,
                "threshold_sigma": self.threshold_sigma,
                "calibration": self._calibration_source,
                "calibration_sequences": self._calibration_sequences,
                "calibration_stats": self._calibration_stats,
                "anomaly_count": int(labels.sum()),
                "mean_reconstruction_error": float(point_errors.mean()),
            },
        )
