# Руководство по запуску

## 1. Python

Проект поддерживает Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
ruff check .
pytest -v
```

## 2. DVC и локальный MinIO

```powershell
docker compose up -d minio createbucket
$env:AWS_ACCESS_KEY_ID = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY = "minioadmin"
dvc remote modify --local minio endpointurl http://localhost:9000
dvc push data/reference.dvc eye_cnn_best_val_final.pth.dvc
dvc pull
dvc repro train
dvc status train
```

Файл `.dvc/config.local` и credentials не коммитятся.

## 3. Docker Compose

```powershell
docker compose up --build
docker compose ps
Invoke-RestMethod http://localhost:8000/ready
```

## 4. Обучение и drift

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
python scripts/register_bootstrap.py
python -m backend.src.train --config configs/train.yaml --fast-dev-run
python -m backend.src.drift --reference data/reference --current data/incoming --output reports/drift --params params.yaml
dvc repro drift
```

## 5. Minikube / Argo CD

Установите Docker Desktop, `kubectl` и `minikube`, затем:

```powershell
minikube start --cpus 4 --memory 6144
$token = Read-Host "GitHub PAT" -AsSecureString
.\scripts\bootstrap-minikube.ps1 `
  -GitHubUsername "your-user" `
  -GitHubToken $token `
  -RepositoryUrl "https://github.com/your-user/your-repository.git"
```

Для приватного репозитория PAT требует `repo` и `read:packages`.

Для полностью локального GitOps-прогона без GitHub:

```powershell
.\scripts\bootstrap-minikube-local.ps1
```

Скрипт собирает SHA-образы, загружает их в Minikube, создаёт временный bare Git
repository, запускает `git daemon` и подключает Argo CD к ветке `gitops`.
Для локальной приёмки в snapshot включаются текущие tracked и untracked
неигнорируемые изменения; основной Git repository при этом не коммитится.

После успешного bootstrap полный lifecycle проверяется командой:

```powershell
.\scripts\smoke-minikube.ps1
```

Проверка загружает 20 размеченных изображений, вызывает drift, ожидает
автоматический retrain Job, promotion в MLflow и hot reload модели.
Bootstrap-скрипт также публикует второй локальный GitOps commit и проверяет Argo
CD self-heal и безопасное пересоздание идемпотентного bootstrap hook.

```powershell
kubectl get application -n argocd mlops-eyes
kubectl get all,pvc -n mlops-eyes
kubectl port-forward svc/backend-service 8000:8000 -n mlops-eyes
kubectl port-forward svc/frontend-service 8501:8501 -n mlops-eyes
kubectl port-forward svc/mlflow-service 5001:5000 -n mlops-eyes
kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-eyes
kubectl port-forward svc/grafana-service 3000:3000 -n mlops-eyes
```

## 6. Диагностика

```powershell
kubectl describe application -n argocd mlops-eyes
kubectl logs deployment/backend -n mlops-eyes
kubectl logs job/bootstrap-model -n mlops-eyes
kubectl get jobs,cronjobs -n mlops-eyes
dvc status
```
