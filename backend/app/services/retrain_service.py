"""Local subprocess and Kubernetes Job retraining orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.app.services.metrics_service import (
    record_retrain_request,
    record_retrain_result,
)
from backend.app.storage import (
    latest_retrain_at,
    list_retrain_events,
    save_alert,
    save_retrain_event,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RETRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "retrain.py"
CURRENT_DATASET = Path(os.getenv("CURRENT_DATASET", "data/incoming"))
MIN_EPOCHS = 1
MAX_EPOCHS = 3
AUTO_RETRAIN_MINIMUM_LABELED = int(os.getenv("AUTO_RETRAIN_MINIMUM_LABELED", "20"))
AUTO_RETRAIN_COOLDOWN_SECONDS = int(os.getenv("AUTO_RETRAIN_COOLDOWN_SECONDS", "3600"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_mode() -> str:
    configured = os.getenv("RETRAIN_MODE", "auto")
    if configured != "auto":
        return configured
    return "kubernetes" if os.getenv("KUBERNETES_SERVICE_HOST") else "local"


def count_labeled_current_samples() -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return sum(
        1
        for label in ("opened", "closed")
        for path in (CURRENT_DATASET / label).glob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def auto_retrain_eligibility() -> tuple[bool, str]:
    labeled = count_labeled_current_samples()
    if labeled < AUTO_RETRAIN_MINIMUM_LABELED:
        return (
            False,
            f"auto retrain requires {AUTO_RETRAIN_MINIMUM_LABELED} labeled samples; "
            f"found {labeled}",
        )
    previous = latest_retrain_at("auto")
    if previous:
        elapsed = (datetime.now(timezone.utc) - previous).total_seconds()
        if elapsed < AUTO_RETRAIN_COOLDOWN_SECONDS:
            remaining = int(AUTO_RETRAIN_COOLDOWN_SECONDS - elapsed)
            return False, f"auto retrain cooldown is active for {remaining} seconds"
    return True, "eligible"


def _save_result(
    *,
    job_id: str,
    status: str,
    message: str,
    mode: str,
    trigger_type: str,
    epochs: int,
) -> None:
    save_retrain_event(
        job_id=job_id,
        status=status,
        message=message,
        mode=mode,
        trigger_type=trigger_type,
        epochs=epochs,
        created_at=_now(),
    )
    record_retrain_result(status)
    save_alert(
        level="info" if status == "completed" else "error",
        category="retrain",
        message=message,
        details={"job_id": job_id, "status": status},
    )


def _run_local(job_id: str, epochs: int, trigger_type: str) -> None:
    command = [sys.executable, str(RETRAIN_SCRIPT), "--epochs", str(epochs)]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            _save_result(
                job_id=job_id,
                status="completed",
                message=f"Retraining completed ({epochs} epoch(s))",
                mode="local",
                trigger_type=trigger_type,
                epochs=epochs,
            )
        else:
            detail = (result.stderr or result.stdout or "unknown error").strip()[-1000:]
            _save_result(
                job_id=job_id,
                status="failed",
                message=f"Retraining failed: {detail}",
                mode="local",
                trigger_type=trigger_type,
                epochs=epochs,
            )
    except Exception as exc:  # noqa: BLE001
        _save_result(
            job_id=job_id,
            status="failed",
            message=f"Retraining failed: {exc}",
            mode="local",
            trigger_type=trigger_type,
            epochs=epochs,
        )


def _create_kubernetes_job(job_id: str, epochs: int, trigger_type: str) -> None:
    from kubernetes import client, config

    config.load_incluster_config()
    namespace = os.getenv("K8S_NAMESPACE", "mlops-eyes")
    image = os.getenv("RETRAIN_IMAGE", "mlopseyes-backend:latest")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
    body: dict[str, Any] = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_id,
            "labels": {
                "app": "mlops-eyes-retrain",
                "trigger": trigger_type,
            },
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "metadata": {"labels": {"app": "mlops-eyes-retrain"}},
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": "ghcr-pull-secret"}],
                    "initContainers": [
                        {
                            "name": "bootstrap-weights",
                            "image": image,
                            "command": ["/bin/sh", "-c"],
                            "args": [
                                "dvc remote modify --local minio endpointurl "
                                "http://minio-service:9000 && "
                                "until dvc pull eye_cnn_best_val_final.pth.dvc; "
                                "do sleep 10; done && "
                                "cp eye_cnn_best_val_final.pth "
                                "/model/eye_cnn_best_val_final.pth"
                            ],
                            "env": [
                                {
                                    "name": "AWS_ACCESS_KEY_ID",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "minio-credentials",
                                            "key": "MINIO_ROOT_USER",
                                        }
                                    },
                                },
                                {
                                    "name": "AWS_SECRET_ACCESS_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "minio-credentials",
                                            "key": "MINIO_ROOT_PASSWORD",
                                        }
                                    },
                                },
                            ],
                            "volumeMounts": [
                                {"name": "model", "mountPath": "/model"},
                            ],
                        }
                    ],
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "command": [
                                "python",
                                "scripts/retrain.py",
                                "--epochs",
                                str(epochs),
                            ],
                            "env": [
                                {"name": "MLFLOW_TRACKING_URI", "value": tracking_uri},
                                {
                                    "name": "MLFLOW_REGISTERED_MODEL",
                                    "value": "open-eyes-cnn",
                                },
                                {"name": "MLFLOW_MODEL_ALIAS", "value": "champion"},
                                {
                                    "name": "MODEL_WEIGHTS_PATH",
                                    "value": "/model/eye_cnn_best_val_final.pth",
                                },
                            ],
                            "volumeMounts": [
                                {"name": "data", "mountPath": "/app/data"},
                                {
                                    "name": "model",
                                    "mountPath": "/model",
                                    "readOnly": True,
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "250m", "memory": "512Mi"},
                                "limits": {"cpu": "1", "memory": "2Gi"},
                            },
                        }
                    ],
                    "volumes": [
                        {
                            "name": "data",
                            "persistentVolumeClaim": {"claimName": "mlops-data"},
                        },
                        {"name": "model", "emptyDir": {}},
                    ],
                },
            },
        },
    }
    client.BatchV1Api().create_namespaced_job(namespace=namespace, body=body)


def schedule_retrain(
    epochs: int,
    *,
    trigger_type: str = "manual",
) -> dict[str, Any]:
    if not MIN_EPOCHS <= epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between {MIN_EPOCHS} and {MAX_EPOCHS}")
    if not RETRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"retrain script not found: {RETRAIN_SCRIPT}")
    mode = _runtime_mode()
    job_id = f"retrain-{uuid.uuid4().hex[:12]}"
    message = f"{trigger_type} retraining scheduled for {epochs} epoch(s)"
    if mode == "kubernetes":
        _create_kubernetes_job(job_id, epochs, trigger_type)
    elif mode != "local":
        raise RuntimeError(f"unsupported RETRAIN_MODE: {mode}")
    save_retrain_event(
        job_id=job_id,
        status="started",
        message=message,
        mode=mode,
        trigger_type=trigger_type,
        epochs=epochs,
        created_at=_now(),
    )
    record_retrain_request(trigger_type, mode)
    if mode == "local":
        thread = threading.Thread(
            target=_run_local,
            args=(job_id, epochs, trigger_type),
            name=job_id,
            daemon=True,
        )
        thread.start()
    return {
        "status": "started",
        "message": message,
        "mode": mode,
        "job_id": job_id,
        "trigger_type": trigger_type,
        "epochs": epochs,
    }


def sync_kubernetes_job_statuses() -> None:
    if _runtime_mode() != "kubernetes":
        return
    try:
        from kubernetes import client, config

        config.load_incluster_config()
        api = client.BatchV1Api()
        namespace = os.getenv("K8S_NAMESPACE", "mlops-eyes")
        events = list_retrain_events(100)
        latest: dict[str, dict[str, Any]] = {}
        for event in events:
            if event["job_id"] and event["job_id"] not in latest:
                latest[event["job_id"]] = event
        for job_id, event in latest.items():
            if event["status"] != "started":
                continue
            status = api.read_namespaced_job_status(job_id, namespace).status
            if status.succeeded:
                _save_result(
                    job_id=job_id,
                    status="completed",
                    message="Kubernetes retraining Job completed",
                    mode="kubernetes",
                    trigger_type=event["trigger_type"],
                    epochs=event["epochs"] or 2,
                )
            elif status.failed:
                _save_result(
                    job_id=job_id,
                    status="failed",
                    message="Kubernetes retraining Job failed",
                    mode="kubernetes",
                    trigger_type=event["trigger_type"],
                    epochs=event["epochs"] or 2,
                )
    except Exception:  # noqa: BLE001 - status endpoint must remain available
        return


def load_auto_retrain_settings(path: str | Path = "params.yaml") -> None:
    """Apply params.yaml defaults unless explicitly set through environment."""
    global AUTO_RETRAIN_COOLDOWN_SECONDS, AUTO_RETRAIN_MINIMUM_LABELED
    params_path = Path(path)
    if not params_path.is_file():
        return
    values = (yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}).get("drift", {})
    if "AUTO_RETRAIN_MINIMUM_LABELED" not in os.environ:
        AUTO_RETRAIN_MINIMUM_LABELED = int(
            values.get("auto_retrain_minimum_labeled_samples", 20)
        )
    if "AUTO_RETRAIN_COOLDOWN_SECONDS" not in os.environ:
        AUTO_RETRAIN_COOLDOWN_SECONDS = int(
            values.get("auto_retrain_cooldown_seconds", 3600)
        )


load_auto_retrain_settings()
