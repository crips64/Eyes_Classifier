"""Kustomize, Argo CD, storage, and automation manifest checks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "k8s" / "base"


def test_required_kubernetes_resources_exist():
    for relative in [
        "kustomization.yaml",
        "storage.yaml",
        "minio.yaml",
        "mlflow.yaml",
        "backend.yaml",
        "frontend.yaml",
        "monitoring.yaml",
        "automation.yaml",
        "rbac.yaml",
    ]:
        assert (BASE / relative).is_file()


def test_backend_has_dvc_init_probes_and_pvc():
    content = (BASE / "backend.yaml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "Dockerfile.backend").read_text(encoding="utf-8")
    for value in [
        "dvc pull",
        "/dvc-pointers/reference.dvc",
        "readinessProbe",
        "livenessProbe",
        "mlops-data",
        "RETRAIN_MODE",
        "bootstrap-model",
    ]:
        assert value in content
    assert "dvc.config.local" in dockerfile
    assert "no_scm = true" in (ROOT / "docker" / "dvc.config.local").read_text(
        encoding="utf-8"
    )


def test_drift_cronjob_and_argocd():
    assert "*/10 * * * *" in (BASE / "automation.yaml").read_text(encoding="utf-8")
    application = (ROOT / "k8s" / "argocd" / "application.yaml").read_text(
        encoding="utf-8"
    )
    assert "targetRevision: gitops" in application
    assert "selfHeal: true" in application


def test_local_and_remote_minikube_overlays_are_portable():
    local = ROOT / "k8s" / "overlays" / "minikube-local" / "kustomization.yaml"
    remote = ROOT / "k8s" / "overlays" / "minikube" / "kustomization.yaml"
    assert local.is_file()
    assert "mlopseyes-backend" in local.read_text(encoding="utf-8")
    remote_text = remote.read_text(encoding="utf-8")
    assert "your-github-owner" in remote_text
    assert "artembotsman" not in remote_text.lower()
