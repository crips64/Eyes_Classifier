"""Pydantic schemas for the public MLOps API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EyeLabel = Literal["opened", "closed"]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    storage_ready: bool
    mlflow_reachable: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    prediction_id: int
    score: float
    label: EyeLabel
    true_label: EyeLabel | None = None
    is_anomaly: bool
    created_at: str
    model_version: str


class PredictionRecord(PredictResponse):
    filename: str
    stored_path: str | None = None
    is_correct: bool | None = None


class PredictionsListResponse(BaseModel):
    predictions: list[PredictionRecord] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class DriftRunResponse(BaseModel):
    status: str
    message: str | None = None
    report_id: str | None = None
    report_path: str | None = None
    summary: dict[str, Any] | None = None
    model_version: str | None = None
    auto_retrain_job_id: str | None = None


class DriftLatestResponse(BaseModel):
    status: str
    message: str | None = None
    report: dict[str, Any] | None = None


class DriftHistoryResponse(BaseModel):
    reports: list[dict[str, Any]] = Field(default_factory=list)


class RetrainRequest(BaseModel):
    epochs: int = Field(default=2, ge=1, le=3)


class RetrainResponse(BaseModel):
    status: str
    message: str
    mode: str
    job_id: str | None = None
    trigger_type: str = "manual"
    epochs: int | None = None


class RetrainEvent(BaseModel):
    id: int
    job_id: str | None = None
    status: str
    message: str
    mode: str
    trigger_type: str
    epochs: int | None = None
    model_version: str | None = None
    created_at: str


class RetrainStatusResponse(BaseModel):
    events: list[RetrainEvent] = Field(default_factory=list)


class ExperimentsResponse(BaseModel):
    status: str
    message: str | None = None
    experiments: list[dict[str, Any]] | None = None
    runs: list[dict[str, Any]] | None = None


class ModelsResponse(BaseModel):
    status: str
    message: str | None = None
    models: list[dict[str, Any]] | None = None


class AlertRecord(BaseModel):
    id: int
    level: str
    category: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool
    created_at: str


class AlertsResponse(BaseModel):
    alerts: list[AlertRecord] = Field(default_factory=list)
