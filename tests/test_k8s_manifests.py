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
        "argocd.argoproj.io/hook: Sync",
        "argocd.argoproj.io/hook-delete-policy: BeforeHookCreation",
    ]:
        assert value in content
    assert "dvc.config.local" in dockerfile
    assert "no_scm = true" in (ROOT / "docker" / "dvc.config.local").read_text(
        encoding="utf-8"
    )
    storage = (BASE / "storage.yaml").read_text(encoding="utf-8")
    assert storage.count('argocd.argoproj.io/sync-wave: "-4"') == 4
    minio = (BASE / "minio.yaml").read_text(encoding="utf-8")
    mlflow = (BASE / "mlflow.yaml").read_text(encoding="utf-8")
    assert 'argocd.argoproj.io/sync-wave: "-3"' in minio
    assert 'argocd.argoproj.io/sync-wave: "-2"' in mlflow
    assert minio.count("argocd.argoproj.io/hook: Sync") == 1
    assert "argocd.argoproj.io/hook-delete-policy: BeforeHookCreation" in minio


def test_drift_cronjob_and_argocd():
    assert "*/10 * * * *" in (BASE / "automation.yaml").read_text(encoding="utf-8")
    application = (ROOT / "k8s" / "argocd" / "application.yaml").read_text(
        encoding="utf-8"
    )
    assert "targetRevision: gitops" in application
    assert "selfHeal: true" in application


def test_monitoring_alerts_cover_each_drift_type():
    compose_alerts = (ROOT / "monitoring" / "alerts.yml").read_text(encoding="utf-8")
    cluster_monitoring = (BASE / "monitoring.yaml").read_text(encoding="utf-8")
    expected = (
        "max(mlops_data_drift_detected) > 0 or "
        "max(mlops_target_drift_detected) > 0 or "
        "max(mlops_prediction_drift_detected) > 0 or "
        "max(mlops_concept_drift_detected) > 0"
    )
    assert expected in compose_alerts
    assert expected in cluster_monitoring
    assert not list((ROOT / "k8s" / "monitoring").glob("*.yaml"))


def test_mlflow_2_server_uses_supported_flags():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    manifest = (BASE / "mlflow.yaml").read_text(encoding="utf-8")
    assert "ghcr.io/mlflow/mlflow:v2.22.0" in compose
    assert "ghcr.io/mlflow/mlflow:v2.22.0" in manifest
    assert "--allowed-hosts" not in compose
    assert "--allowed-hosts" not in manifest


def test_local_and_remote_minikube_overlays_are_portable():
    local = ROOT / "k8s" / "overlays" / "minikube-local" / "kustomization.yaml"
    remote = ROOT / "k8s" / "overlays" / "minikube" / "kustomization.yaml"
    assert local.is_file()
    assert "mlopseyes-backend" in local.read_text(encoding="utf-8")
    remote_text = remote.read_text(encoding="utf-8")
    assert "your-github-owner" in remote_text
    assert "artembotsman" not in remote_text.lower()

    bootstrap = (ROOT / "scripts" / "bootstrap-minikube-local.ps1").read_text(
        encoding="utf-8"
    )
    assert "VerifyRepeatedSync" in bootstrap
    assert "IncludeWorkingTree" in bootstrap
    assert "mlops-eyes/repeated-sync" in bootstrap
    assert "Invoke-Dvc" in bootstrap
    assert "Argo CD did not create the MinIO deployment" in bootstrap
    assert "--server-side --force-conflicts" in bootstrap
    assert "emptyDockerConfig" in bootstrap

    smoke = (ROOT / "scripts" / "smoke-minikube.ps1").read_text(encoding="utf-8")
    assert "Registered model diagnostics" in smoke
    assert "Retrain Job logs" in smoke
    assert "Repeated Argo CD sync did not reach Synced / Healthy" in bootstrap
