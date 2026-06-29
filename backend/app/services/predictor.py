"""Validated image ingestion and atomically reloadable model inference."""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from backend.app.services.metrics_service import set_active_model
from open_eyes_classifier import OpenEyesClassificator

BOOTSTRAP_VERSION = "bootstrap"
DEFAULT_WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", "eye_cnn_best_val_final.pth")
INCOMING_DATASET = Path(os.getenv("CURRENT_DATASET", "data/incoming"))
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "open-eyes-cnn")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "30"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
LOW_BRIGHTNESS_THRESHOLD = 30.0
HIGH_BRIGHTNESS_THRESHOLD = 225.0
UNCERTAIN_LOW = 0.4
UNCERTAIN_HIGH = 0.6
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class Predictor(Protocol):
    def predict(self, image_path: str) -> float: ...


class PytorchModelPredictor:
    def __init__(self, model: torch.nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [transforms.Grayscale(), transforms.Resize((24, 24)), transforms.ToTensor()]
        )

    def predict(self, image_path: str) -> float:
        image = Image.open(image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(tensor).item())


class ModelManager:
    """Keeps inference available while a new MLflow model is downloaded."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS):
        self.weights_path = weights_path
        self._predictor: Predictor | None = None
        self._version = BOOTSTRAP_VERSION
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_refresh_error: str | None = None

    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._predictor is not None

    def load_bootstrap(self) -> None:
        predictor = OpenEyesClassificator(weights_path=self.weights_path)
        with self._lock:
            self._predictor = predictor
            self._version = f"{BOOTSTRAP_VERSION}:{Path(self.weights_path).name}"
            self.last_refresh_error = None
        set_active_model(self._version)

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load_bootstrap()

    def predict(self, image_path: str) -> tuple[float, str]:
        self.ensure_loaded()
        with self._lock:
            if self._predictor is None:  # pragma: no cover - guarded by ensure_loaded
                raise RuntimeError("model is not loaded")
            predictor = self._predictor
            version = self._version
        return predictor.predict(image_path), version

    def snapshot(self) -> tuple[Predictor, str]:
        """Return one immutable predictor/version pair for batch evaluation."""
        self.ensure_loaded()
        with self._lock:
            if self._predictor is None:  # pragma: no cover - guarded by ensure_loaded
                raise RuntimeError("model is not loaded")
            return self._predictor, self._version

    def refresh_from_registry(self) -> bool:
        """Load the champion outside the lock and swap it only after success."""
        try:
            import mlflow
            import mlflow.pytorch
            from mlflow.tracking import MlflowClient

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            candidate = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
            candidate_version = f"mlflow:{REGISTERED_MODEL_NAME}:{candidate.version}"
            if candidate_version == self.version:
                return False
            model = mlflow.pytorch.load_model(
                f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}",
                map_location="cpu",
            )
            predictor = PytorchModelPredictor(model)
            with self._lock:
                self._predictor = predictor
                self._version = candidate_version
                self.last_refresh_error = None
            set_active_model(candidate_version)
            return True
        except Exception as exc:  # noqa: BLE001 - registry is optional at bootstrap
            self.last_refresh_error = str(exc)
            return False

    def _refresh_loop(self) -> None:
        while not self._stop.wait(MODEL_REFRESH_SECONDS):
            self.refresh_from_registry()

    def start(self) -> None:
        self.ensure_loaded()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._refresh_loop,
                name="model-registry-refresh",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


model_manager = ModelManager()


def score_to_label(score: float) -> str:
    return "opened" if score >= 0.5 else "closed"


def _mean_brightness(image_path: str) -> float:
    image = Image.open(image_path).convert("L").resize((24, 24))
    return float(np.asarray(image, dtype=np.float32).mean())


def detect_anomaly(score: float, image_path: str) -> bool:
    uncertain = UNCERTAIN_LOW <= score <= UNCERTAIN_HIGH
    brightness = _mean_brightness(image_path)
    return uncertain or brightness < LOW_BRIGHTNESS_THRESHOLD or brightness > HIGH_BRIGHTNESS_THRESHOLD


def _validate_image(filename: str, content_type: str | None, content: bytes) -> str:
    if not content:
        raise ValueError("uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"uploaded file exceeds {MAX_UPLOAD_BYTES} bytes")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("only .jpg, .jpeg and .png files are supported")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("unsupported image content type")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("uploaded file is not a valid image") from exc
    return suffix


def ingest_and_predict(
    *,
    filename: str,
    content_type: str | None,
    content: bytes,
    true_label: str | None,
) -> tuple[float, str, bool, str, str]:
    suffix = _validate_image(filename, content_type, content)
    bucket = true_label if true_label in {"opened", "closed"} else "unlabeled"
    destination_dir = INCOMING_DATASET / bucket
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{int(time.time())}-{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(content)
    try:
        score, model_version = model_manager.predict(str(destination))
        label = score_to_label(score)
        anomaly = detect_anomaly(score, str(destination))
        return score, label, anomaly, str(destination), model_version
    except Exception:
        destination.unlink(missing_ok=True)
        raise
