"""FastAPI service for inference and the complete course MLOps lifecycle."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from backend.app.schemas import (
    AlertsResponse,
    DriftHistoryResponse,
    DriftLatestResponse,
    DriftRunResponse,
    ExperimentsResponse,
    HealthResponse,
    ModelsResponse,
    PredictionRecord,
    PredictionsListResponse,
    PredictResponse,
    ReadyResponse,
    RetrainEvent,
    RetrainRequest,
    RetrainResponse,
    RetrainStatusResponse,
)
from backend.app.services.drift_service import REPORT_DIR, get_latest_drift_report, run_drift_report
from backend.app.services.metrics_service import (
    record_prediction,
    record_prediction_error,
)
from backend.app.services.mlflow_service import (
    is_mlflow_reachable,
    list_experiments,
    list_registered_models,
)
from backend.app.services.predictor import ingest_and_predict, model_manager
from backend.app.services.retrain_service import schedule_retrain, sync_kubernetes_job_statuses
from backend.app.storage import (
    acknowledge_alert,
    init_db,
    list_alerts,
    list_drift_reports,
    list_predictions,
    list_retrain_events,
    save_alert,
    save_prediction,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    try:
        model_manager.start()
    except Exception as exc:  # noqa: BLE001
        save_alert(
            level="error",
            category="model",
            message=f"Bootstrap model failed to load: {exc}",
        )
    yield
    model_manager.stop()


app = FastAPI(
    title="Open Eyes Classifier API",
    description=(
        "Inference, labeled production data collection, drift monitoring, "
        "MLflow registry, and retraining orchestration"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=model_manager.version)


@app.get("/ready", response_model=ReadyResponse, tags=["operations"])
def ready() -> ReadyResponse:
    storage_ready = True
    storage_error: str | None = None
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001
        storage_ready = False
        storage_error = str(exc)
    model_loaded = model_manager.loaded
    return ReadyResponse(
        ready=model_loaded and storage_ready,
        model_loaded=model_loaded,
        storage_ready=storage_ready,
        mlflow_reachable=is_mlflow_reachable(),
        details={
            "model_version": model_manager.version,
            "model_refresh_error": model_manager.last_refresh_error,
            "storage_error": storage_error,
        },
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(...),
    true_label: Literal["opened", "closed"] | None = Form(default=None),
) -> PredictResponse:
    started = time.perf_counter()
    content = await file.read()
    try:
        score, label, is_anomaly, stored_path, model_version = ingest_and_predict(
            filename=file.filename or "upload.jpg",
            content_type=file.content_type,
            content=content,
            true_label=true_label,
        )
    except ValueError as exc:
        record_prediction_error("validation")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        record_prediction_error("model_unavailable")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        record_prediction_error("inference")
        raise HTTPException(status_code=500, detail="inference failed") from exc

    created_at = datetime.now(timezone.utc).isoformat()
    prediction_id = save_prediction(
        filename=file.filename or "upload.jpg",
        stored_path=stored_path,
        score=score,
        label=label,
        true_label=true_label,
        is_anomaly=is_anomaly,
        created_at=created_at,
        model_version=model_version,
    )
    record_prediction(
        score=score,
        label=label,
        true_label=true_label,
        is_anomaly=is_anomaly,
        latency_seconds=time.perf_counter() - started,
    )
    return PredictResponse(
        prediction_id=prediction_id,
        score=round(score, 4),
        label=label,
        true_label=true_label,
        is_anomaly=is_anomaly,
        created_at=created_at,
        model_version=model_version,
    )


@app.get("/predictions", response_model=PredictionsListResponse, tags=["inference"])
def predictions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    label: Literal["opened", "closed"] | None = None,
    anomaly: bool | None = None,
) -> PredictionsListResponse:
    rows, total = list_predictions(
        limit=limit,
        offset=offset,
        label=label,
        anomaly=anomaly,
    )
    return PredictionsListResponse(
        predictions=[PredictionRecord(**row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.post("/drift/run", response_model=DriftRunResponse, tags=["drift"])
def drift_run() -> DriftRunResponse:
    return DriftRunResponse(**run_drift_report())


@app.get("/drift/latest", response_model=DriftLatestResponse, tags=["drift"])
def drift_latest() -> DriftLatestResponse:
    report = get_latest_drift_report()
    if report is None:
        return DriftLatestResponse(
            status="not_available",
            message="drift report not generated yet",
        )
    return DriftLatestResponse(status="available", report=report)


@app.get("/drift/history", response_model=DriftHistoryResponse, tags=["drift"])
def drift_history(limit: int = Query(default=20, ge=1, le=100)) -> DriftHistoryResponse:
    return DriftHistoryResponse(reports=list_drift_reports(limit))


@app.post("/retrain", response_model=RetrainResponse, tags=["retraining"])
def retrain(body: RetrainRequest | None = None) -> RetrainResponse:
    request = body or RetrainRequest()
    try:
        return RetrainResponse(**schedule_retrain(request.epochs, trigger_type="manual"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/retrain/status", response_model=RetrainStatusResponse, tags=["retraining"])
def retrain_status(
    limit: int = Query(default=20, ge=1, le=100),
) -> RetrainStatusResponse:
    sync_kubernetes_job_statuses()
    rows = list_retrain_events(limit)
    return RetrainStatusResponse(events=[RetrainEvent(**row) for row in rows])


@app.get("/experiments", response_model=ExperimentsResponse, tags=["mlflow"])
def experiments() -> ExperimentsResponse:
    return ExperimentsResponse(**list_experiments())


@app.get("/models", response_model=ModelsResponse, tags=["mlflow"])
def models() -> ModelsResponse:
    return ModelsResponse(**list_registered_models())


@app.get("/alerts", response_model=AlertsResponse, tags=["operations"])
def alerts(
    limit: int = Query(default=50, ge=1, le=200),
    unacknowledged_only: bool = False,
) -> AlertsResponse:
    return AlertsResponse(
        alerts=list_alerts(limit=limit, unacknowledged_only=unacknowledged_only)
    )


@app.post("/alerts/{alert_id}/acknowledge", tags=["operations"])
def acknowledge(alert_id: int) -> dict[str, bool]:
    if not acknowledge_alert(alert_id):
        raise HTTPException(status_code=404, detail="alert not found")
    return {"acknowledged": True}


@app.get("/reports/{report_id}", tags=["drift"])
def report_file(report_id: str, format: Literal["json", "html"] = "html") -> Response:
    if not report_id.startswith("drift-") or any(char in report_id for char in "/\\"):
        raise HTTPException(status_code=400, detail="invalid report id")
    path = Path(REPORT_DIR) / f"{report_id}.{format}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    media_type = "application/json" if format == "json" else "text/html"
    return Response(path.read_bytes(), media_type=media_type)


@app.get("/metrics", tags=["operations"])
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
