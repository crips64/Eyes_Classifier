# Contributing

## Git flow

The project uses GitHub Flow:

1. Create a short-lived branch from `main`.
2. Add tests and documentation with the change.
3. Open a pull request; CI must pass.
4. Squash-merge the pull request into `main`.
5. CI publishes immutable images and updates the `gitops` branch. Argo CD deploys it.

Direct feature commits to `main` are not part of the workflow.

## Conventional Commits

Commit and pull-request titles use:

```text
<type>(optional-scope): <description>
```

Allowed types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
`ci`, and `chore`.

Examples:

```text
feat(api): persist labeled inference images
fix(drift): apply thresholds from params.yaml
ci(gitops): publish immutable image tags
```

## Required local checks

```bash
ruff check .
pytest -v
docker compose config --quiet
kubectl kustomize k8s/overlays/minikube
```
