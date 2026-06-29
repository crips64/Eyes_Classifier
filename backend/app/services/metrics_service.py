"""Prometheus metrics for API, model quality, drift, and retraining."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter("mlops_predictions_total", "Predictions served")
PREDICTIONS_BY_LABEL = Counter(
    "mlops_predictions_by_label_total",
    "Predictions by label",
    ["label"],
)
PREDICTION_ERRORS_TOTAL = Counter(
    "mlops_prediction_errors_total",
    "Prediction failures",
    ["reason"],
)
ANOMALY_PREDICTIONS_TOTAL = Counter(
    "mlops_anomaly_predictions_total",
    "Anomaly predictions",
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "mlops_prediction_latency_seconds",
    "Prediction latency",
)
PREDICTION_SCORE = Histogram(
    "mlops_prediction_score",
    "Prediction score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
LABELED_PREDICTIONS_TOTAL = Counter(
    "mlops_labeled_predictions_total",
    "Predictions submitted with ground truth",
)
CORRECT_LABELED_PREDICTIONS_TOTAL = Counter(
    "mlops_correct_labeled_predictions_total",
    "Correct predictions with ground truth",
)
LABELED_ACCURACY = Gauge(
    "mlops_labeled_accuracy",
    "Running accuracy for labeled production samples",
)
DRIFT_RUNS_TOTAL = Counter("mlops_drift_runs_total", "Drift report runs")
DRIFT_ALERTS_TOTAL = Counter("mlops_drift_alerts_total", "Drift alerts")
DATA_DRIFT_DETECTED = Gauge("mlops_data_drift_detected", "Data drift flag")
TARGET_DRIFT_DETECTED = Gauge("mlops_target_drift_detected", "Target drift flag")
PREDICTION_DRIFT_DETECTED = Gauge(
    "mlops_prediction_drift_detected",
    "Predicted-label distribution drift flag",
)
CONCEPT_DRIFT_DETECTED = Gauge("mlops_concept_drift_detected", "Concept drift flag")
CONCEPT_ACCURACY = Gauge("mlops_concept_accuracy", "Accuracy on labeled current data")
RETRAIN_REQUESTS_TOTAL = Counter(
    "mlops_retrain_requests_total",
    "Retraining requests",
    ["trigger", "mode"],
)
RETRAIN_RESULTS_TOTAL = Counter(
    "mlops_retrain_results_total",
    "Retraining outcomes",
    ["status"],
)
ACTIVE_MODEL = Gauge(
    "mlops_active_model",
    "Active model version (the active label has value 1)",
    ["version"],
)

_labeled_count = 0
_correct_count = 0
_active_version: str | None = None


def restore_labeled_accuracy(labeled_count: int, correct_count: int) -> None:
    """Restore the persisted business-quality gauge after a backend restart."""
    global _correct_count, _labeled_count
    _labeled_count = max(0, labeled_count)
    _correct_count = min(max(0, correct_count), _labeled_count)
    LABELED_ACCURACY.set(
        _correct_count / _labeled_count if _labeled_count else 0
    )


def record_prediction(
    *,
    score: float,
    label: str,
    true_label: str | None,
    is_anomaly: bool,
    latency_seconds: float,
) -> None:
    global _correct_count, _labeled_count
    PREDICTIONS_TOTAL.inc()
    PREDICTIONS_BY_LABEL.labels(label=label).inc()
    if is_anomaly:
        ANOMALY_PREDICTIONS_TOTAL.inc()
    PREDICTION_LATENCY_SECONDS.observe(latency_seconds)
    PREDICTION_SCORE.observe(score)
    if true_label:
        _labeled_count += 1
        LABELED_PREDICTIONS_TOTAL.inc()
        if true_label == label:
            _correct_count += 1
            CORRECT_LABELED_PREDICTIONS_TOTAL.inc()
        LABELED_ACCURACY.set(_correct_count / _labeled_count)


def record_prediction_error(reason: str) -> None:
    PREDICTION_ERRORS_TOTAL.labels(reason=reason).inc()


def record_drift_report(report: dict) -> None:
    DRIFT_RUNS_TOTAL.inc()
    data_flag = bool(report.get("data_drift", {}).get("data_drift_detected", False))
    target_flag = bool(report.get("target_drift", {}).get("target_drift_detected", False))
    prediction_flag = bool(
        report.get("prediction_drift", {}).get("prediction_drift_detected", False)
    )
    concept = report.get("concept_drift", {})
    concept_flag = bool(concept.get("concept_drift_detected", False))
    DATA_DRIFT_DETECTED.set(int(data_flag))
    TARGET_DRIFT_DETECTED.set(int(target_flag))
    PREDICTION_DRIFT_DETECTED.set(int(prediction_flag))
    CONCEPT_DRIFT_DETECTED.set(int(concept_flag))
    if concept.get("accuracy") is not None:
        CONCEPT_ACCURACY.set(float(concept["accuracy"]))
    if data_flag or target_flag or prediction_flag or concept_flag:
        DRIFT_ALERTS_TOTAL.inc()


def record_retrain_request(trigger: str, mode: str) -> None:
    RETRAIN_REQUESTS_TOTAL.labels(trigger=trigger, mode=mode).inc()


def record_retrain_result(status: str) -> None:
    RETRAIN_RESULTS_TOTAL.labels(status=status).inc()


def set_active_model(version: str) -> None:
    global _active_version
    if _active_version:
        ACTIVE_MODEL.labels(version=_active_version).set(0)
    ACTIVE_MODEL.labels(version=version).set(1)
    _active_version = version
