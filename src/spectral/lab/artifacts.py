"""Versioned LSTM artefact persistence.

An artefact is a directory containing a CPU ``state_dict`` and three JSON
documents: architecture/training metadata, the normalisation parameters, and
the calibration parameters. Loading validates all of them and restores the
detector without ever calling ``fit``.

No Python pickle outside the tensor ``state_dict`` is loaded: the model is
reconstructed from JSON configuration and ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from spectral.detectors.lstm_autoencoder import (
    LSTMAutoencoderDetector,
    _LSTMAutoencoder,
)
from spectral.lab.data import fingerprint_frame

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "lstm_autoencoder"
REQUIRED_FILES = ("model.pt", "metadata.json", "preprocessing.json", "calibration.json")
_MODEL_DIR_RE = re.compile(r"model_(\d+)")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ArtifactError(RuntimeError):
    """Raised for missing, corrupt or incompatible model artefacts."""


@dataclass(frozen=True)
class LoadedLSTMModel:
    """A loaded CPU detector plus the documents that describe it."""

    detector: LSTMAutoencoderDetector
    path: Path
    metadata: dict
    preprocessing: dict
    calibration: dict

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", self.path.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path.name} must contain a JSON object")
    return payload


def next_model_version(root: str | Path) -> str:
    """Return the next ``model_NNN`` version available under ``root``."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in root_path.iterdir():
        match = _MODEL_DIR_RE.fullmatch(child.name)
        if match and child.is_dir():
            numbers.append(int(match.group(1)))
    return f"model_{max(numbers, default=0) + 1:03d}"


def list_artifacts(root: str | Path) -> list[Path]:
    """Model directories under ``root`` that contain a metadata document."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(
        child
        for child in root_path.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    )


def delete_lstm_artifact(root: str | Path, model_version: str) -> Path:
    """Delete one artefact directory under ``root`` after validating its name.

    Only a direct child of ``root`` whose name matches the version pattern and
    that looks like an artefact (contains ``metadata.json``) can be removed.
    This is the workbench's delete path; it never accepts an arbitrary path.
    """
    root_path = Path(root)
    if not _VERSION_RE.fullmatch(str(model_version)):
        raise ArtifactError("invalid model_version for deletion")
    target = (root_path / model_version).resolve()
    if target.parent != root_path.resolve():
        raise ArtifactError("refusing to delete a path outside the artefact root")
    if not target.is_dir() or not (target / "metadata.json").is_file():
        raise ArtifactError(f"artefact not found: {target}")
    shutil.rmtree(target)
    return target


def save_lstm_artifact(
    detector: LSTMAutoencoderDetector,
    root: str | Path,
    *,
    model_version: str | None = None,
    dataset_frame=None,
    train_points: int | None = None,
    evaluation: dict | None = None,
    fit_seconds: float | None = None,
    extra_metadata: dict | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a fitted detector to ``root/<model_version>`` and return the path.

    The model is always written from CPU tensors. ``dataset_frame`` stores a
    fingerprint so later evaluation can prove which data produced the model.
    """
    if not detector._fitted or detector._model is None or detector._threshold is None:
        raise ArtifactError("detector must be fitted before it can be saved")
    if detector._threshold <= 0 or not np.isfinite(detector._threshold):
        raise ArtifactError("detector has an invalid calibration threshold")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    version = model_version or next_model_version(root_path)
    if not _VERSION_RE.fullmatch(version):
        raise ArtifactError(
            "model_version may contain only letters, digits, dot, underscore and dash"
        )
    target = root_path / version
    if target.exists() and not overwrite:
        raise ArtifactError(f"artefact already exists: {target}")

    temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-model-", dir=root_path))
    try:
        state = {
            key: value.detach().to("cpu")
            for key, value in detector._model.state_dict().items()
        }
        model_path = temp_dir / "model.pt"
        torch.save(state, model_path)
        checksum = _sha256(model_path)

        dataset_metadata: dict = {}
        if dataset_frame is not None:
            dataset_metadata = {
                "fingerprint_sha256": fingerprint_frame(dataset_frame),
                "n_points": int(len(dataset_frame)),
                "timestamp_start": str(dataset_frame["timestamp"].iloc[0]),
                "timestamp_end": str(dataset_frame["timestamp"].iloc[-1]),
            }

        architecture = {
            "input_dim": 1,
            "hidden_dim": int(detector.hidden_dim),
            "n_layers": int(detector.n_layers),
            "sequence_length": int(detector.sequence_length),
        }
        training = {
            "epochs": int(detector.epochs),
            "lr": float(detector.lr),
            "batch_size": int(detector.batch_size),
            "patience": int(detector.patience),
            "threshold_sigma": float(detector.threshold_sigma),
            "seed": int(detector.seed),
            "calibration_fraction": float(detector.calibration_fraction),
            "device": str(detector._device),
            "train_points": int(train_points) if train_points is not None else None,
            "calibration_source": detector._calibration_source,
            "calibration_windows": int(detector._calibration_sequences),
            "fit_seconds": float(fit_seconds) if fit_seconds is not None else None,
            "history": list(getattr(detector, "_training_history", [])),
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "model_version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "framework": {"name": "pytorch", "version": torch.__version__},
            "architecture": architecture,
            "training": training,
            "dataset": dataset_metadata,
            "evaluation": evaluation or {},
            "model_sha256": checksum,
            "model_bytes": int(model_path.stat().st_size),
        }
        if extra_metadata:
            metadata["extra"] = extra_metadata

        preprocessing = {
            "schema_version": SCHEMA_VERSION,
            "normalization": "zscore",
            "train_mean": float(detector._train_mean),
            "train_std": float(detector._train_std),
        }
        calibration = {
            "schema_version": SCHEMA_VERSION,
            "threshold": float(detector._threshold),
            "threshold_sigma": float(detector.threshold_sigma),
            "source": detector._calibration_source,
            "windows": int(detector._calibration_sequences),
            "stats": detector._calibration_stats,
        }

        _write_json(temp_dir / "metadata.json", metadata)
        _write_json(temp_dir / "preprocessing.json", preprocessing)
        _write_json(temp_dir / "calibration.json", calibration)

        if target.exists():
            if not overwrite:
                raise ArtifactError(f"artefact already exists: {target}")
            shutil.rmtree(target)
        temp_dir.rename(target)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return target


def load_lstm_artifact(directory: str | Path) -> LoadedLSTMModel:
    """Validate and load an artefact on CPU without training."""
    path = Path(directory)
    if not path.is_dir():
        raise ArtifactError(f"artefact directory not found: {path}")

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ArtifactError(
            f"artefact {path} is incomplete; missing: {', '.join(missing)}"
        )

    metadata = _read_json(path / "metadata.json")
    preprocessing = _read_json(path / "preprocessing.json")
    calibration = _read_json(path / "calibration.json")

    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"unsupported artefact schema {metadata.get('schema_version')!r}; "
            f"this build supports {SCHEMA_VERSION}"
        )
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactError(
            f"artefact type {metadata.get('artifact_type')!r} is not {ARTIFACT_TYPE}"
        )

    architecture = metadata.get("architecture")
    training = metadata.get("training")
    if not isinstance(architecture, dict) or not isinstance(training, dict):
        raise ArtifactError("metadata is missing architecture or training configuration")

    try:
        hidden_dim = int(architecture["hidden_dim"])
        n_layers = int(architecture["n_layers"])
        sequence_length = int(architecture["sequence_length"])
        train_mean = float(preprocessing["train_mean"])
        train_std = float(preprocessing["train_std"])
        threshold = float(calibration["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"artefact configuration is malformed: {exc}") from exc

    if hidden_dim <= 0 or n_layers <= 0 or sequence_length < 2:
        raise ArtifactError("artefact architecture values must be positive")
    if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std <= 0:
        raise ArtifactError("artefact normalisation parameters are invalid")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ArtifactError("artefact calibration threshold is invalid")

    expected_checksum = metadata.get("model_sha256")
    actual_checksum = _sha256(path / "model.pt")
    if expected_checksum != actual_checksum:
        raise ArtifactError(
            "model.pt checksum does not match metadata; the artefact is corrupt"
        )

    try:
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated exception types
        raise ArtifactError(f"cannot load model.pt: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise ArtifactError("model.pt does not contain a state_dict")

    detector = LSTMAutoencoderDetector(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        sequence_length=sequence_length,
        epochs=int(training.get("epochs", 50)),
        lr=float(training.get("lr", 1e-3)),
        batch_size=int(training.get("batch_size", 32)),
        threshold_sigma=float(
            calibration.get("threshold_sigma", training.get("threshold_sigma", 2.0))
        ),
        patience=int(training.get("patience", 5)),
        seed=int(training.get("seed", 42)),
        calibration_fraction=float(training.get("calibration_fraction", 0.2)),
    )
    # CPU-first is deliberate: the model is tiny, and loading must behave the
    # same on Apple Silicon, Linux CI and machines without CUDA.
    detector._device = torch.device("cpu")
    model = _LSTMAutoencoder(
        input_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
    ).to(detector._device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ArtifactError(f"model state is incompatible with metadata: {exc}") from exc
    model.eval()

    detector._model = model
    detector._train_mean = train_mean
    detector._train_std = train_std
    detector._threshold = threshold
    detector._calibration_source = calibration.get("source")
    detector._calibration_sequences = int(calibration.get("windows", 0))
    detector._calibration_stats = dict(calibration.get("stats") or {})
    detector._training_history = list(training.get("history") or [])
    detector._fitted = True

    return LoadedLSTMModel(
        detector=detector,
        path=path,
        metadata=metadata,
        preprocessing=preprocessing,
        calibration=calibration,
    )
