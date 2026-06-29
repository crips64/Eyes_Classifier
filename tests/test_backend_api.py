"""API tests using deterministic temporary model weights and storage."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.services.metrics_service import restore_labeled_accuracy
from backend.app.services.predictor import ModelManager, registry_model_uri
from backend.app.storage import init_db, labeled_prediction_stats, save_prediction
from open_eyes_classifier import MediumEyeCNN


@pytest.fixture()
def client(tmp_path, monkeypatch):
    weights = tmp_path / "model.pth"
    torch.manual_seed(42)
    torch.save(MediumEyeCNN().state_dict(), weights)
    manager = ModelManager(str(weights))
    database = tmp_path / "predictions.db"
    incoming = tmp_path / "incoming"
    monkeypatch.setattr("backend.app.storage.DEFAULT_DB_PATH", database)
    monkeypatch.setattr("backend.app.main.model_manager", manager)
    monkeypatch.setattr("backend.app.services.predictor.model_manager", manager)
    monkeypatch.setattr("backend.app.services.predictor.INCOMING_DATASET", incoming)
    init_db(database)
    with TestClient(app) as test_client:
        yield test_client


def _image_bytes(value: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (24, 24), color=value).save(buffer, format="PNG")
    return buffer.getvalue()


def test_registry_model_uri_uses_proxy_for_legacy_server_path():
    source = "/mlflow/artifacts/1/run-id/artifacts/model"

    assert (
        registry_model_uri(source, "open-eyes-cnn", "champion")
        == "mlflow-artifacts:/1/run-id/artifacts/model"
    )
    assert (
        registry_model_uri("s3://bucket/model", "open-eyes-cnn", "champion")
        == "models:/open-eyes-cnn@champion"
    )


def test_health_and_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"].startswith("bootstrap:")
    assert client.get("/ready").json()["ready"] is True


def test_predict_persists_labeled_image(client):
    response = client.post(
        "/predict",
        files={"file": ("tiny.png", _image_bytes(), "image/png")},
        data={"true_label": "opened"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction_id"] > 0
    assert 0 <= payload["score"] <= 1
    assert payload["true_label"] == "opened"
    assert payload["model_version"].startswith("bootstrap:")

    predictions = client.get("/predictions?label=opened&limit=10").json()
    assert predictions["total"] == 1
    assert predictions["predictions"][0]["prediction_id"] == payload["prediction_id"]
    all_predictions = client.get("/predictions?limit=10").json()
    assert all_predictions["total"] == 1
    assert Path(all_predictions["predictions"][0]["stored_path"]).is_file()


def test_predict_rejects_invalid_file(client):
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlops_predictions_total" in response.text
    assert "mlops_active_model" in response.text


def test_drift_latest_when_report_missing(client, monkeypatch):
    monkeypatch.setattr("backend.app.main.get_latest_drift_report", lambda: None)
    response = client.get("/drift/latest")
    assert response.status_code == 200
    assert response.json()["status"] == "not_available"


def test_report_download_uses_configured_directory(client, tmp_path, monkeypatch):
    report_id = "drift-20260628T000000Z"
    report = tmp_path / f"{report_id}.html"
    report.write_text("<h1>drift</h1>", encoding="utf-8")
    monkeypatch.setattr("backend.app.main.REPORT_DIR", str(tmp_path))

    response = client.get(f"/reports/{report_id}?format=html")

    assert response.status_code == 200
    assert response.text == "<h1>drift</h1>"


def test_retrain_endpoint(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.schedule_retrain",
        lambda epochs, trigger_type: {
            "status": "started",
            "message": "scheduled",
            "mode": "local",
            "job_id": "retrain-test",
            "trigger_type": trigger_type,
            "epochs": epochs,
        },
    )
    payload = client.post("/retrain", json={"epochs": 2}).json()
    assert payload["job_id"] == "retrain-test"
    assert payload["trigger_type"] == "manual"


def test_experiments_and_models_fallback(client, monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.list_experiments",
        lambda: {
            "status": "not_available",
            "message": "MLflow tracking server is not available",
            "experiments": None,
            "runs": None,
        },
    )
    monkeypatch.setattr(
        "backend.app.main.list_registered_models",
        lambda: {
            "status": "not_available",
            "message": "MLflow tracking server is not available",
            "models": None,
        },
    )
    assert client.get("/experiments").json()["status"] == "not_available"
    assert client.get("/models").json()["status"] == "not_available"


def test_alert_lifecycle(client):
    from backend.app.storage import save_alert

    alert_id = save_alert(level="warning", category="test", message="test alert")
    response = client.get("/alerts?unacknowledged_only=true")
    assert any(alert["id"] == alert_id for alert in response.json()["alerts"])
    assert client.post(f"/alerts/{alert_id}/acknowledge").status_code == 200


def test_labeled_accuracy_is_restored_from_storage(tmp_path):
    database = tmp_path / "predictions.db"
    init_db(database)
    for index, (label, true_label) in enumerate(
        [("opened", "opened"), ("closed", "opened"), ("closed", None)]
    ):
        save_prediction(
            filename=f"{index}.png",
            stored_path=f"/tmp/{index}.png",
            score=0.5,
            label=label,
            true_label=true_label,
            is_anomaly=False,
            created_at="2026-06-29T00:00:00+00:00",
            model_version="test",
            db_path=database,
        )

    labeled, correct = labeled_prediction_stats(database)
    restore_labeled_accuracy(labeled, correct)

    assert (labeled, correct) == (2, 1)
    assert "mlops_labeled_accuracy 0.5" in client_metrics_text()


def client_metrics_text() -> str:
    from prometheus_client import generate_latest

    return generate_latest().decode()
