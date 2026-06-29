# Drift and monitoring

`params.yaml` is the source of drift thresholds.

- data drift: KS tests over image statistics and histogram bins;
- target drift: delta in the true-label distribution for labeled samples;
- prediction drift: delta in the active model's predicted-label distribution;
- concept drift: accuracy/F1 on current images with `true_label`.

The API uses the currently loaded MLflow model for prediction and concept drift.
Reports include the exact `model_version`; target drift is explicitly marked
`not_available` when current labels are missing.

Reports are saved as timestamped JSON/HTML files and as `latest_*`. Their
metadata and alerts are persisted in SQLite.

Prometheus exposes inference volume/latency/errors, anomaly and class counts,
labeled accuracy, drift flags, retrain outcomes and active model version.
Grafana is provisioned automatically. The combined alert uses separate PromQL
comparisons for data, target, prediction and concept drift, so a target-only or
concept-only event cannot be hidden by another zero-valued series.

The running labeled accuracy is rebuilt from persisted predictions on backend
startup instead of resetting to an empty process-local value.

The Kubernetes CronJob calls `POST /drift/run` every 10 minutes. A warning starts
automatic retraining only when at least 20 current samples are labeled and the
one-hour cooldown has elapsed.
