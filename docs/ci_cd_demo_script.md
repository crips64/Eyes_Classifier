# CI/CD demonstration

## Pull request

Show a Conventional PR title and the following successful jobs:

- Ruff;
- pytest with coverage;
- DVC pipeline validation without remote access;
- Kustomize and Docker Compose validation;
- backend and frontend Docker builds.

## Merge to main

Show two private GHCR packages with both `latest` and immutable
`sha-<commit>` tags. Then show the bot commit in the `gitops` branch.

## Argo CD

Show that the Application tracks `gitops`, is `Synced / Healthy`, and that the
running backend/frontend pod image IDs correspond to the published commit.

```bash
kubectl get application -n argocd mlops-eyes
kubectl get pods -n mlops-eyes -o jsonpath="{..image}"
```
