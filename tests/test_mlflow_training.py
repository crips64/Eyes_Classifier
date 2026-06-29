"""Tests for MLflow training pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def test_train_module_exists():
    assert (ROOT / "backend" / "src" / "train.py").is_file()


def test_train_config_exists():
    assert (ROOT / "configs" / "train.yaml").is_file()


def test_train_config_parse():
    config = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    assert config["experiment_name"] == "open-eyes-classifier"
    assert config["registered_model_name"] == "open-eyes-cnn"
    assert config["dataset_path"] == "data/reference"
    assert config["epochs"] >= 1


def test_build_train_config_from_yaml():
    from backend.src.train import TrainConfig, build_train_config

    class Args:
        config = str(ROOT / "configs" / "train.yaml")
        data = None
        additional_data = None
        epochs = None
        batch_size = None
        learning_rate = None
        experiment_name = None
        registered_model_name = None
        fast_dev_run = True

    config = build_train_config(Args())
    assert isinstance(config, TrainConfig)
    assert config.fast_dev_run is True
    assert config.epochs == 1
    assert config.experiment_name == "open-eyes-classifier"


def test_fast_dev_training_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    from backend.src.train import TrainConfig, train_model

    dataset = tmp_path / "dataset"
    for label, value in (("opened", 200), ("closed", 40)):
        for index in range(8):
            path = dataset / label / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (24, 24), color=value + index).save(path)

    config = TrainConfig(
        data=str(dataset),
        additional_data=None,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        fast_dev_run=True,
        model_output_path=str(tmp_path / "model.pth"),
        minimum_accuracy=0.0,
    )
    summary = train_model(config)
    assert summary["run_id"]
    assert "accuracy" in summary["metrics"]
    assert Path(summary["weights_path"]).exists()


def test_list_experiments_returns_not_available_on_timeout(monkeypatch):
    import time

    from backend.app.services import mlflow_service

    class SlowClient:
        def search_experiments(self):
            time.sleep(10)
            return []

    monkeypatch.setattr(mlflow_service, "MLFLOW_REQUEST_TIMEOUT", 0.1)
    monkeypatch.setattr(mlflow_service, "_client", lambda: SlowClient())

    started = time.perf_counter()
    result = mlflow_service.list_experiments()
    elapsed = time.perf_counter() - started

    assert result["status"] == "not_available"
    assert elapsed < 2


def test_retrain_falls_back_to_bootstrap_weights(tmp_path, monkeypatch):
    from backend.src.train import TrainConfig, initialize_model
    from open_eyes_classifier import MediumEyeCNN

    weights = tmp_path / "bootstrap.pth"
    torch.save(MediumEyeCNN().state_dict(), weights)
    monkeypatch.setattr(
        "backend.src.train.mlflow.pytorch.load_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no champion")),
    )
    config = TrainConfig(
        initialize_from_champion=True,
        initial_weights_path=str(weights),
    )
    model, source = initialize_model(config, torch.device("cpu"))
    assert isinstance(model, MediumEyeCNN)
    assert source == f"weights:{weights}"


def test_bootstrap_registration_is_idempotent():
    from scripts.register_bootstrap import existing_champion_version

    champion = type("Version", (), {"version": "7"})()
    client = type(
        "Client",
        (),
        {"get_model_version_by_alias": lambda self, name, alias: champion},
    )()
    assert existing_champion_version(client, "model", "champion") == "7"

    class MissingAlias:
        def get_model_version_by_alias(self, name, alias):
            raise MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)

    assert existing_champion_version(MissingAlias(), "model", "champion") is None

    class MissingAliasOnMlflow222:
        def get_model_version_by_alias(self, name, alias):
            raise MlflowException(
                "Registered model alias champion not found",
                error_code=INVALID_PARAMETER_VALUE,
            )

    assert (
        existing_champion_version(MissingAliasOnMlflow222(), "model", "champion")
        is None
    )
