"""Timeout-bounded MLflow read APIs used by FastAPI and readiness checks."""

from __future__ import annotations

import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
from math import ceil
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, TypeVar

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_REQUEST_TIMEOUT = float(os.getenv("MLFLOW_REQUEST_TIMEOUT", "5"))
T = TypeVar("T")


def _client():
    # MLflow otherwise retries an unavailable server for several minutes. These
    # settings also bound the daemon worker after the public API has timed out.
    os.environ.setdefault(
        "MLFLOW_HTTP_REQUEST_TIMEOUT",
        str(max(1, ceil(MLFLOW_REQUEST_TIMEOUT))),
    )
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def _call_with_timeout(func: Callable[[], T]) -> T:
    result: Queue[tuple[bool, T | BaseException]] = Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((True, func()))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            result.put((False, exc))

    Thread(target=run, name="mlflow-api-call", daemon=True).start()
    try:
        succeeded, value = result.get(timeout=MLFLOW_REQUEST_TIMEOUT)
    except Empty as exc:
        raise FuturesTimeoutError from exc
    if succeeded:
        return value  # type: ignore[return-value]
    raise value  # type: ignore[misc]


def _not_available(kind: str) -> dict[str, Any]:
    return {
        "status": "not_available",
        "message": "MLflow tracking server is not available",
        kind: None,
    }


def is_mlflow_reachable() -> bool:
    try:
        _call_with_timeout(lambda: _client().search_experiments(max_results=1))
        return True
    except Exception:  # noqa: BLE001
        return False


def list_experiments() -> dict[str, Any]:
    try:
        client = _client()
        experiments = _call_with_timeout(lambda: client.search_experiments())
        experiment_payload = [
            {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "lifecycle_stage": experiment.lifecycle_stage,
                "artifact_location": experiment.artifact_location,
            }
            for experiment in experiments
        ]
        experiment_ids = [item["experiment_id"] for item in experiment_payload]
        runs = (
            _call_with_timeout(
                lambda: client.search_runs(
                    experiment_ids=experiment_ids,
                    order_by=["attributes.start_time DESC"],
                    max_results=50,
                )
            )
            if experiment_ids
            else []
        )
        run_payload = [
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": {
                    key: value
                    for key, value in run.data.tags.items()
                    if key.startswith("mlops.") or key == "mlflow.runName"
                },
            }
            for run in runs
        ]
        return {"status": "ok", "experiments": experiment_payload, "runs": run_payload}
    except (FuturesTimeoutError, Exception):  # noqa: BLE001
        result = _not_available("experiments")
        result["runs"] = None
        return result


def list_registered_models() -> dict[str, Any]:
    try:
        client = _client()
        models = _call_with_timeout(lambda: client.search_registered_models())
        payload = []
        for model in models:
            versions = _call_with_timeout(
                lambda name=model.name: client.search_model_versions(f"name='{name}'")
            )
            payload.append(
                {
                    "name": model.name,
                    "versions": [
                        {
                            "version": version.version,
                            "status": version.status,
                            "run_id": version.run_id,
                            "aliases": list(getattr(version, "aliases", []) or []),
                            "creation_timestamp": version.creation_timestamp,
                        }
                        for version in versions
                    ],
                }
            )
        return {"status": "ok", "models": payload}
    except (FuturesTimeoutError, Exception):  # noqa: BLE001
        return _not_available("models")
