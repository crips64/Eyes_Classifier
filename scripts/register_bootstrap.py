#!/usr/bin/env python3
"""Register the DVC bootstrap weights as the first MLflow champion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torchvision import transforms

from backend.src.train import (
    EyesDatasetLoader,
    TrainConfig,
    collect_predictions,
    compute_metrics,
    register_and_maybe_promote,
    split_dataset,
)
from open_eyes_classifier import MediumEyeCNN


def existing_champion_version(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> str | None:
    """Return the active version, while only treating a missing alias as absent."""
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        message = str(exc).lower()
        if (
            exc.error_code == "INVALID_PARAMETER_VALUE"
            and "alias" in message
            and "not found" in message
        ):
            return None
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Register bootstrap model in MLflow")
    parser.add_argument("--weights", default="eye_cnn_best_val_final.pth")
    parser.add_argument("--data", default="data/reference")
    args = parser.parse_args()
    if not Path(args.weights).is_file() or not Path(args.data).is_dir():
        print("Bootstrap weights or reference dataset are not available")
        return 1

    config = TrainConfig(data=args.data)
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    champion_version = existing_champion_version(
        MlflowClient(),
        config.registered_model_name,
        config.promotion_alias,
    )
    if champion_version is not None:
        print(
            f"Champion already exists: model={config.registered_model_name} "
            f"version={champion_version}; bootstrap registration skipped"
        )
        return 0

    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((24, 24)), transforms.ToTensor()]
    )
    dataset = EyesDatasetLoader(args.data, transform)
    _, validation = split_dataset(dataset, config.validation_ratio, config.seed)
    loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MediumEyeCNN().to(device)
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )
    model.eval()
    y_true, y_pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred)

    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="bootstrap-model") as run:
        mlflow.log_params(
            {
                "source": "dvc-bootstrap",
                "validation_ratio": config.validation_ratio,
                "seed": config.seed,
            }
        )
        mlflow.log_metrics(metrics)
        model_info = mlflow.pytorch.log_model(model, artifact_path="model")
        promotion = register_and_maybe_promote(
            model_uri=model_info.model_uri,
            run_id=run.info.run_id,
            accuracy=metrics["accuracy"],
            config=config,
        )
    print(
        f"Bootstrap model version={promotion['version']} "
        f"promoted={promotion['promoted']} accuracy={metrics['accuracy']:.4f}"
    )
    return 0 if promotion["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
