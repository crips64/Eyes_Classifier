"""Drift calculation and timestamped report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from backend.src.drift import (
    DriftConfig,
    build_drift_report,
    compute_prediction_drift,
    compute_target_drift,
)
from backend.src.features import extract_dataset_features, extract_image_features
from open_eyes_classifier import MediumEyeCNN


def _save_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (24, 24), color=value).save(path)


def _weights(path: Path) -> Path:
    torch.manual_seed(42)
    torch.save(MediumEyeCNN().state_dict(), path)
    return path


def test_feature_extraction(tmp_path):
    image = tmp_path / "sample.png"
    _save_image(image, 120)
    features = extract_image_features(str(image))
    assert features["mean_pixel"] == 120
    assert "hist_7" in features


def test_extract_dataset_features_with_labels(tmp_path):
    _save_image(tmp_path / "opened" / "a.jpg", 200)
    _save_image(tmp_path / "closed" / "b.jpg", 40)
    frame = extract_dataset_features(str(tmp_path))
    assert set(frame["label"]) == {"opened", "closed"}


def test_drift_report_and_history_files(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    output = tmp_path / "reports"
    _save_image(reference / "opened" / "r1.jpg", 180)
    _save_image(reference / "closed" / "r2.jpg", 60)
    _save_image(current / "opened" / "c1.jpg", 170)
    _save_image(current / "closed" / "c2.jpg", 55)
    report = build_drift_report(
        str(reference),
        str(current),
        str(output),
        str(_weights(tmp_path / "weights.pth")),
        max_samples_per_dataset=10,
        config=DriftConfig(minimum_samples=2),
    )
    assert report["report_id"].startswith("drift-")
    assert (output / "latest_drift_report.json").is_file()
    assert (output / f"{report['report_id']}.html").is_file()
    saved = json.loads((output / "latest_drift_report.json").read_text(encoding="utf-8"))
    assert {"data_drift", "target_drift", "concept_drift"} <= saved.keys()


def test_not_enough_data_report(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    _save_image(reference / "opened" / "r.jpg", 180)
    current.mkdir()
    report = build_drift_report(
        str(reference),
        str(current),
        str(tmp_path / "reports"),
        str(tmp_path / "missing-not-used.pth"),
        config=DriftConfig(minimum_samples=1),
    )
    assert report["status"] == "not_enough_data"


def test_target_and_prediction_drift_are_distinct():
    reference = pd.DataFrame(
        {
            "true_label": ["opened", "closed"],
            "predicted_label": ["opened", "closed"],
        }
    )
    current = pd.DataFrame(
        {
            "true_label": ["opened", "closed"],
            "predicted_label": ["opened", "opened"],
        }
    )
    target = compute_target_drift(reference, current, threshold=0.2)
    prediction = compute_prediction_drift(reference, current, threshold=0.2)
    assert target["target_drift_status"] == "available"
    assert target["target_drift_detected"] is False
    assert prediction["prediction_drift_detected"] is True


def test_report_uses_injected_active_predictor(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    for root in (reference, current):
        _save_image(root / "opened" / "opened.jpg", 180)
        _save_image(root / "closed" / "closed.jpg", 60)

    calls: list[str] = []

    def predictor(path: str) -> float:
        calls.append(path)
        return 0.9 if "opened" in path else 0.1

    report = build_drift_report(
        str(reference),
        str(current),
        str(tmp_path / "reports"),
        str(tmp_path / "unused.pth"),
        config=DriftConfig(minimum_samples=2),
        predictor=predictor,
        model_version="mlflow:open-eyes-cnn:7",
    )
    assert report["model_version"] == "mlflow:open-eyes-cnn:7"
    assert report["concept_drift"]["accuracy"] == 1.0
    assert len(calls) == 4
