"""Timestamped HTML/JSON drift report generation."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def _render_features_table(features: list[dict[str, Any]]) -> str:
    if not features:
        return "<p>No feature statistics available.</p>"
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('feature', '')))}</td>"
        f"<td>{float(item.get('statistic', 0)):.4f}</td>"
        f"<td>{float(item.get('p_value', 0)):.6f}</td>"
        f"<td>{escape(str(item.get('drift_detected', False)))}</td>"
        "</tr>"
        for item in features
    )
    return (
        "<table><thead><tr><th>Feature</th><th>KS statistic</th>"
        f"<th>p-value</th><th>Drift</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_drift_html(report: dict[str, Any]) -> str:
    data = report.get("data_drift", {})
    target = report.get("target_drift", {})
    prediction = report.get("prediction_drift", {})
    concept = report.get("concept_drift", {})
    warning = report.get("status") == "warning"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Drift Report {escape(str(report.get("report_id", "")))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 7px; text-align: left; }}
    .ok {{ color: #15803d; font-weight: bold; }}
    .warning {{ color: #b91c1c; font-weight: bold; }}
    .box {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>Open Eyes Drift Report</h1>
  <p><strong>Report:</strong> {escape(str(report.get("report_id", "")))}</p>
  <p><strong>Generated:</strong> {escape(str(report.get("generated_at", "")))}</p>
  <p><strong>Model:</strong> {escape(str(report.get("model_version", "unknown")))}</p>
  <p class="{"warning" if warning else "ok"}"><strong>Status:</strong>
    {escape(str(report.get("overall_status", report.get("status", "unknown"))))}
  </p>
  <p>Reference samples: {report.get("reference_samples", 0)} ·
     Current samples: {report.get("current_samples", 0)}</p>
  <div class="box"><h2>Data drift</h2>
    <p>Detected: {data.get("data_drift_detected", False)}</p>
    {_render_features_table(data.get("features", []))}
  </div>
  <div class="box"><h2>Target drift</h2>
    <p>Status: {target.get("target_drift_status", "unknown")}</p>
    <p>Reference opened ratio: {target.get("reference_opened_ratio")}</p>
    <p>Current opened ratio: {target.get("current_opened_ratio")}</p>
    <p>Delta: {target.get("delta")}</p>
    <p>Detected: {target.get("target_drift_detected", False)}</p>
  </div>
  <div class="box"><h2>Prediction drift</h2>
    <p>Status: {prediction.get("prediction_drift_status", "unknown")}</p>
    <p>Reference opened ratio: {prediction.get("reference_opened_ratio")}</p>
    <p>Current opened ratio: {prediction.get("current_opened_ratio")}</p>
    <p>Delta: {prediction.get("delta")}</p>
    <p>Detected: {prediction.get("prediction_drift_detected", False)}</p>
  </div>
  <div class="box"><h2>Concept drift</h2>
    <p>Status: {concept.get("concept_drift_status", "unknown")}</p>
    <p>Reason: {escape(str(concept.get("reason", "-")))}</p>
    <p>Accuracy: {concept.get("accuracy", "-")}</p>
    <p>Precision: {concept.get("precision", "-")}</p>
    <p>Recall: {concept.get("recall", "-")}</p>
    <p>F1: {concept.get("f1", "-")}</p>
    <p>Detected: {concept.get("concept_drift_detected", False)}</p>
  </div>
</body>
</html>
"""


def save_drift_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_id = str(report["report_id"])
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    html_text = render_drift_html(report)
    paths = {
        "json": output / f"{report_id}.json",
        "html": output / f"{report_id}.html",
        "latest_json": output / "latest_drift_report.json",
        "latest_html": output / "latest_drift_report.html",
    }
    paths["json"].write_text(json_text, encoding="utf-8")
    paths["html"].write_text(html_text, encoding="utf-8")
    paths["latest_json"].write_text(json_text, encoding="utf-8")
    paths["latest_html"].write_text(html_text, encoding="utf-8")
    return paths
