"""Kubernetes retraining must be independent from the active backend pod."""

from __future__ import annotations

import sys
from types import SimpleNamespace


def test_kubernetes_retrain_job_pulls_bootstrap_weights(monkeypatch):
    from backend.app.services import retrain_service

    captured: dict = {}

    class BatchApi:
        def create_namespaced_job(self, *, namespace, body):
            captured["namespace"] = namespace
            captured["body"] = body

    fake_kubernetes = SimpleNamespace(
        client=SimpleNamespace(BatchV1Api=lambda: BatchApi()),
        config=SimpleNamespace(load_incluster_config=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "kubernetes", fake_kubernetes)
    monkeypatch.setenv("RETRAIN_IMAGE", "mlopseyes-backend:sha-test")

    retrain_service._create_kubernetes_job("retrain-test", 2, "auto")

    pod = captured["body"]["spec"]["template"]["spec"]
    init = pod["initContainers"][0]
    trainer = pod["containers"][0]
    assert captured["namespace"] == "mlops-eyes"
    assert init["image"] == "mlopseyes-backend:sha-test"
    assert "dvc pull eye_cnn_best_val_final.pth.dvc" in init["args"][0]
    assert any(item["name"] == "minio-credentials" for item in [
        env["valueFrom"]["secretKeyRef"] for env in init["env"]
    ])
    assert {
        "name": "MODEL_WEIGHTS_PATH",
        "value": "/model/eye_cnn_best_val_final.pth",
    } in trainer["env"]
    assert {"name": "model", "emptyDir": {}} in pod["volumes"]
