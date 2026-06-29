# Architecture decisions

- Minikube is the production-like course target; Compose is a development environment.
- CI builds code-only images because the selected DVC remote is local MinIO.
- DVC stores the immutable reference dataset and bootstrap weights.
- MLflow stores trained candidates and controls serving through the `champion` alias.
- One FastAPI replica and SQLite on PVC keep the course deployment understandable.
- Retraining runs in a Kubernetes Job and never blocks the API process.
- Automatic retraining requires labels; pseudo-labeling is intentionally excluded.
- Argo CD reads a private `gitops` branch and deploys immutable image tags.
- All credentials are runtime Kubernetes/GitHub secrets and are absent from Git.
