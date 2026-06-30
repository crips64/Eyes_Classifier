# Readiness report

Дата проверки: 2026-06-29.

## Итог

Проект готов к сдаче по локальному Minikube/GitOps контуру. Argo CD разворачивает
стек, DVC-артефакты доступны через MinIO, backend и frontend healthy, drift
создает retrain Job, MLflow registry продвигает champion, FastAPI hot-reload
подхватывает новую модель, Prometheus metrics и Streamlit health доступны.

Ограничение: настоящий GitHub/GHCR контур не прогонялся, потому что для него
нужны PAT/secrets и доступ к удаленному репозиторию. Локальный GitOps-контур
проверен полностью.

## Проверки

| Проверка | Результат |
|---|---|
| `.venv\Scripts\python.exe -m ruff check .` | Passed, `All checks passed!` |
| `.venv\Scripts\python.exe -m pytest -v --cov=backend --cov-fail-under=75` | Passed, `61 passed`, coverage `77.39%` |
| `.venv\Scripts\python.exe -m dvc status train` | Passed, `Data and pipelines are up to date.` |
| `docker compose config --quiet` | Passed |
| `kubectl kustomize k8s\overlays\minikube` | Passed |
| `.\scripts\bootstrap-minikube-local.ps1` | Passed, Argo CD `Synced / Healthy` |
| `.\scripts\smoke-minikube.ps1` | Passed, `End-to-end smoke passed: mlflow:open-eyes-cnn:9 -> mlflow:open-eyes-cnn:10` |

## Требования курса

| Требование | Статус | Артефакты |
|---|---|---|
| Датасет и базовая модель | Готово | `data/reference.dvc`, `eye_cnn_best_val_final.pth.dvc`, `open_eyes_classifier.py` |
| Git flow, conventional commits, DVC | Готово | `CONTRIBUTING.md`, `.github/workflows/ci-cd.yml`, `.dvc/config`, `dvc.yaml` |
| Cookiecutter | Готово | `cookiecutter-mlops-eyes/`, `tests/test_cookiecutter_template.py` |
| MLflow tracking и registry | Готово | `backend/src/train.py`, `scripts/register_bootstrap.py`, `k8s/base/mlflow.yaml` |
| CI/CD | Готово локально, remote не прогонялся | `.github/workflows/ci-cd.yml` |
| FastAPI + OpenAPI + Docker | Готово | `backend/app/main.py`, `docker/Dockerfile.backend`, `docker-compose.yml` |
| Drift monitoring + Prometheus/Grafana | Готово | `backend/src/drift.py`, `monitoring/`, `k8s/base/monitoring.yaml` |
| Drift reports | Готово | `/drift/run`, `/drift/latest`, `/reports/{report_id}` |
| Web UI | Готово | `frontend/streamlit_app.py` |
| Argo CD в Minikube | Готово | `k8s/argocd/application.yaml`, `scripts/bootstrap-minikube-local.ps1` |
| README и запуск | Готово | `README.md`, `docs/run_guide.md`, `DEMO.md` |

## Что было исправлено при доведении

- Kubernetes retrain Job теперь монтирует `mlflow-data` PVC в `/mlflow`, чтобы
  MLflow model artifacts сохранялись там, откуда MLflow server может их отдавать.
- Argo CD bootstrap model Job также монтирует `mlflow-data` PVC в `/mlflow`.
- Локальный bootstrap теперь тегирует образы по snapshot commit, который включает
  dirty working tree, и останавливается при ошибке Docker build/image load.
- Smoke test теперь выбирает свободный локальный port-forward port, если 8000 или
  8501 заняты на машине.
