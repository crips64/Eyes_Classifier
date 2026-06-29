# Cookiecutter template

Generate a new project:

```bash
cookiecutter cookiecutter-mlops-eyes
```

The template includes FastAPI, Streamlit, Docker Compose, DVC/MLflow skeletons,
CI, Kustomize and an Argo CD Application. It intentionally excludes datasets,
weights, credentials and runtime databases.

```bash
cd <project_slug>
pip install -r requirements-dev.txt
pytest -v
docker compose config --quiet
kubectl kustomize k8s/overlays/minikube
```
