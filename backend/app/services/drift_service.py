"""Drift orchestration, report persistence, alerts, and guarded auto retraining."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.app.services.metrics_service import record_drift_report
from backend.app.services.predictor import model_manager
from backend.app.services.retrain_service import (
    auto_retrain_eligibility,
    schedule_retrain,
)
from backend.app.storage import save_alert, save_drift_report_record
from backend.src.drift import build_drift_report, load_drift_config, load_latest_report

REFERENCE_DATASET = os.getenv("REFERENCE_DATASET", "data/reference")
CURRENT_DATASET = os.getenv("CURRENT_DATASET", "data/incoming")
REPORT_DIR = os.getenv("DRIFT_REPORT_DIR", "reports/drift")
WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "eye_cnn_best_val_final.pth")
PARAMS_PATH = os.getenv("PARAMS_PATH", "params.yaml")


def run_drift_report() -> dict[str, Any]:
    active_predictor, active_version = model_manager.snapshot()
    report = build_drift_report(
        reference_path=REFERENCE_DATASET,
        current_path=CURRENT_DATASET,
        output_dir=REPORT_DIR,
        weights_path=WEIGHTS_PATH,
        config=load_drift_config(PARAMS_PATH),
        predictor=active_predictor.predict,
        model_version=active_version,
    )
    record_drift_report(report)
    paths = report.get("report_paths", {})
    save_drift_report_record(
        report_id=report["report_id"],
        status=report["status"],
        json_path=paths.get(
            "json", str(Path(REPORT_DIR) / f"{report['report_id']}.json")
        ),
        html_path=paths.get(
            "html", str(Path(REPORT_DIR) / f"{report['report_id']}.html")
        ),
        summary=report,
        created_at=report["generated_at"],
    )
    auto_job_id: str | None = None
    message = report.get("message")
    if report["status"] == "warning":
        save_alert(
            level="warning",
            category="drift",
            message="Drift detected in production data",
            details={"report_id": report["report_id"], "report": report},
            created_at=report["generated_at"],
        )
        eligible, reason = auto_retrain_eligibility()
        if eligible:
            result = schedule_retrain(2, trigger_type="auto")
            auto_job_id = result["job_id"]
            message = f"Drift detected; automatic retraining started: {auto_job_id}"
        else:
            message = f"Drift detected; automatic retraining skipped: {reason}"
    return {
        "status": report["status"],
        "message": message,
        "report_id": report["report_id"],
        "report_path": paths.get("json"),
        "summary": report,
        "model_version": report.get("model_version"),
        "auto_retrain_job_id": auto_job_id,
    }


def get_latest_drift_report() -> dict[str, Any] | None:
    return load_latest_report(REPORT_DIR)
