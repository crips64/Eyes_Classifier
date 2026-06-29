"""Data, predicted-target, and concept drift detection for eye images."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.src.features import extract_dataset_features
from backend.src.reports import save_drift_report

FEATURE_COLUMNS = [
    "mean_pixel",
    "std_pixel",
    "min_pixel",
    "max_pixel",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    *[f"hist_{index}" for index in range(8)],
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class DriftConfig:
    ks_alpha: float = 0.05
    target_drift_threshold: float = 0.2
    concept_drift_accuracy_threshold: float = 0.85
    minimum_samples: int = 20


def load_drift_config(path: str | Path | None) -> DriftConfig:
    if path is None or not Path(path).is_file():
        return DriftConfig()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    values = payload.get("drift", payload)
    return DriftConfig(
        ks_alpha=float(values.get("ks_alpha", 0.05)),
        target_drift_threshold=float(values.get("target_drift_threshold", 0.2)),
        concept_drift_accuracy_threshold=float(
            values.get("concept_drift_accuracy_threshold", 0.85)
        ),
        minimum_samples=int(values.get("minimum_samples", 20)),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _list_images(dataset_path: Path) -> list[Path]:
    if not dataset_path.exists():
        return []
    return sorted(
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _score_to_label(score: float) -> str:
    return "opened" if score >= 0.5 else "closed"


def _predict_dataset_scores(
    dataset_path: Path,
    weights_path: str,
    max_samples: int | None,
    predictor: Callable[[str], float] | None = None,
) -> pd.DataFrame:
    images = _list_images(dataset_path)
    if max_samples and len(images) > max_samples:
        images = list(pd.Series(images).sample(max_samples, random_state=42))
    if predictor is None:
        from open_eyes_classifier import OpenEyesClassificator

        classifier = OpenEyesClassificator(weights_path=weights_path)
        predictor = classifier.predict
    rows: list[dict[str, Any]] = []
    for image_path in images:
        score = predictor(str(image_path))
        row: dict[str, Any] = {
            "filepath": str(image_path),
            "score": score,
            "predicted_label": _score_to_label(score),
        }
        try:
            first_part = image_path.relative_to(dataset_path).parts[0]
        except (ValueError, IndexError):
            first_part = ""
        if first_part in {"opened", "closed"}:
            row["true_label"] = first_part
        rows.append(row)
    return pd.DataFrame(rows)


def compute_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    feature_results: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        reference = reference_df[feature].dropna().to_numpy()
        current = current_df[feature].dropna().to_numpy()
        if len(reference) == 0 or len(current) == 0:
            continue
        statistic, p_value = stats.ks_2samp(reference, current, method="auto")
        feature_results.append(
            {
                "feature": feature,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < alpha),
            }
        )
    return {
        "alpha": alpha,
        "data_drift_detected": any(item["drift_detected"] for item in feature_results),
        "features": feature_results,
    }


def compute_target_drift(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, Any]:
    if (
        "true_label" not in reference_predictions.columns
        or "true_label" not in current_predictions.columns
    ):
        return {
            "target_drift_status": "not_available",
            "reason": "true labels are required for target drift",
            "reference_opened_ratio": None,
            "current_opened_ratio": None,
            "delta": None,
            "threshold": threshold,
            "target_drift_detected": False,
        }
    reference = reference_predictions.dropna(subset=["true_label"])
    current = current_predictions.dropna(subset=["true_label"])
    if reference.empty or current.empty:
        return {
            "target_drift_status": "not_available",
            "reason": "true labels are required for target drift",
            "reference_opened_ratio": None,
            "current_opened_ratio": None,
            "delta": None,
            "threshold": threshold,
            "target_drift_detected": False,
        }
    reference_ratio = float((reference["true_label"] == "opened").mean())
    current_ratio = float((current["true_label"] == "opened").mean())
    delta = abs(reference_ratio - current_ratio)
    return {
        "target_drift_status": "available",
        "reference_labeled_samples": len(reference),
        "current_labeled_samples": len(current),
        "reference_opened_ratio": reference_ratio,
        "current_opened_ratio": current_ratio,
        "delta": delta,
        "threshold": threshold,
        "target_drift_detected": bool(delta > threshold),
    }


def compute_prediction_drift(
    reference_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    threshold: float = 0.2,
) -> dict[str, Any]:
    if reference_predictions.empty or current_predictions.empty:
        return {
            "prediction_drift_status": "not_available",
            "reference_opened_ratio": None,
            "current_opened_ratio": None,
            "delta": None,
            "threshold": threshold,
            "prediction_drift_detected": False,
        }
    reference_ratio = float((reference_predictions["predicted_label"] == "opened").mean())
    current_ratio = float((current_predictions["predicted_label"] == "opened").mean())
    delta = abs(reference_ratio - current_ratio)
    return {
        "prediction_drift_status": "available",
        "reference_opened_ratio": reference_ratio,
        "current_opened_ratio": current_ratio,
        "delta": delta,
        "threshold": threshold,
        "prediction_drift_detected": bool(delta > threshold),
    }


def compute_concept_drift(
    current_predictions: pd.DataFrame,
    accuracy_threshold: float = 0.85,
) -> dict[str, Any]:
    if "true_label" not in current_predictions.columns:
        return {
            "concept_drift_status": "not_available",
            "reason": "true labels are not available for current data",
            "accuracy_threshold": accuracy_threshold,
            "concept_drift_detected": False,
        }
    labeled = current_predictions.dropna(subset=["true_label"])
    if labeled.empty:
        return {
            "concept_drift_status": "not_available",
            "reason": "true labels are not available for current data",
            "accuracy_threshold": accuracy_threshold,
            "concept_drift_detected": False,
        }
    y_true = labeled["true_label"].tolist()
    y_pred = labeled["predicted_label"].tolist()
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        "concept_drift_status": "available",
        "labeled_samples": len(labeled),
        "accuracy": accuracy,
        "precision": float(
            precision_score(y_true, y_pred, pos_label="opened", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label="opened", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="opened", zero_division=0)),
        "accuracy_threshold": accuracy_threshold,
        "concept_drift_detected": accuracy < accuracy_threshold,
    }


def build_drift_report(
    reference_path: str,
    current_path: str,
    output_dir: str,
    weights_path: str = "eye_cnn_best_val_final.pth",
    max_samples_per_dataset: int | None = 200,
    config: DriftConfig | None = None,
    predictor: Callable[[str], float] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    settings = config or DriftConfig()
    reference = Path(reference_path)
    current = Path(current_path)
    now = _utc_now()
    report_id = f"drift-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    current_images = _list_images(current)
    if len(current_images) < settings.minimum_samples:
        report = {
            "report_id": report_id,
            "generated_at": now.isoformat(),
            "status": "not_enough_data",
            "message": (
                f"At least {settings.minimum_samples} current samples are required; "
                f"found {len(current_images)}"
            ),
            "reference_dataset": str(reference),
            "current_dataset": str(current),
            "reference_samples": len(_list_images(reference)),
            "current_samples": len(current_images),
            "model_version": model_version or f"weights:{Path(weights_path).name}",
            "data_drift": {"data_drift_detected": False, "features": []},
            "target_drift": {
                "target_drift_status": "not_available",
                "target_drift_detected": False,
            },
            "prediction_drift": {
                "prediction_drift_status": "not_available",
                "prediction_drift_detected": False,
            },
            "concept_drift": {
                "concept_drift_status": "not_available",
                "reason": "not enough current data",
                "concept_drift_detected": False,
            },
            "overall_status": "NOT ENOUGH DATA",
        }
        paths = save_drift_report(report, output_dir)
        report["report_paths"] = {key: str(path) for key, path in paths.items()}
        return report

    reference_df = extract_dataset_features(str(reference))
    current_df = extract_dataset_features(str(current))
    if max_samples_per_dataset and len(reference_df) > max_samples_per_dataset:
        reference_df = reference_df.sample(max_samples_per_dataset, random_state=42)
    if max_samples_per_dataset and len(current_df) > max_samples_per_dataset:
        current_df = current_df.sample(max_samples_per_dataset, random_state=42)
    data_drift = compute_data_drift(reference_df, current_df, settings.ks_alpha)
    reference_predictions = _predict_dataset_scores(
        reference, weights_path, max_samples_per_dataset, predictor
    )
    current_predictions = _predict_dataset_scores(
        current, weights_path, max_samples_per_dataset, predictor
    )
    target_drift = compute_target_drift(
        reference_predictions,
        current_predictions,
        settings.target_drift_threshold,
    )
    prediction_drift = compute_prediction_drift(
        reference_predictions,
        current_predictions,
        settings.target_drift_threshold,
    )
    concept_drift = compute_concept_drift(
        current_predictions,
        settings.concept_drift_accuracy_threshold,
    )
    any_drift = any(
        (
            data_drift["data_drift_detected"],
            target_drift["target_drift_detected"],
            prediction_drift["prediction_drift_detected"],
            concept_drift["concept_drift_detected"],
        )
    )
    report = {
        "report_id": report_id,
        "generated_at": now.isoformat(),
        "status": "warning" if any_drift else "ok",
        "reference_dataset": str(reference),
        "current_dataset": str(current),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "model_version": model_version or f"weights:{Path(weights_path).name}",
        "data_drift": data_drift,
        "target_drift": target_drift,
        "prediction_drift": prediction_drift,
        "concept_drift": concept_drift,
        "overall_status": "WARNING: drift detected" if any_drift else "OK",
    }
    paths = save_drift_report(report, output_dir)
    report["report_paths"] = {key: str(path) for key, path in paths.items()}
    return report


def load_latest_report(output_dir: str = "reports/drift") -> dict[str, Any] | None:
    report_path = Path(output_dir) / "latest_drift_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift detection report")
    parser.add_argument("--reference", default="data/reference")
    parser.add_argument("--current", default="data/incoming")
    parser.add_argument("--output", default="reports/drift")
    parser.add_argument("--weights", default="eye_cnn_best_val_final.pth")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    report = build_drift_report(
        args.reference,
        args.current,
        args.output,
        args.weights,
        config=load_drift_config(args.params),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
