"""Deterministic training, MLflow tracking, registry, and champion promotion."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms

from open_eyes_classifier import MediumEyeCNN

plt.switch_backend("Agg")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class TrainConfig:
    data: str = "data/reference"
    additional_data: str | None = "data/incoming"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    validation_ratio: float = 0.2
    experiment_name: str = "open-eyes-classifier"
    registered_model_name: str = "open-eyes-cnn"
    model_output_path: str = "artifacts/models/open_eyes_cnn_mlflow.pth"
    promotion_alias: str = "champion"
    minimum_accuracy: float = 0.85
    maximum_accuracy_drop: float = 0.01
    initialize_from_champion: bool = False
    initial_weights_path: str | None = None
    fast_dev_run: bool = False


class EyesDatasetLoader(Dataset):
    def __init__(
        self,
        roots: str | list[str],
        transform: transforms.Compose,
        max_per_class: int | None = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, float]] = []
        root_list = [roots] if isinstance(roots, str) else roots
        for root in root_list:
            root_path = Path(root)
            for label_name, label_value in (("opened", 1.0), ("closed", 0.0)):
                class_dir = root_path / label_name
                if not class_dir.exists():
                    continue
                images = sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                if max_per_class is not None:
                    images = images[:max_per_class]
                self.samples.extend((path, label_value) for path in images)
        if not self.samples:
            raise FileNotFoundError(f"No training images found under {root_list}")

    @property
    def labels(self) -> list[int]:
        return [int(label) for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor([label], dtype=torch.float32)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if getattr(args, "config", None):
        values = load_config(args.config)
        promotion = values.get("promotion", {})
        config.data = str(values.get("dataset_path", config.data))
        config.additional_data = values.get(
            "additional_dataset_path", config.additional_data
        )
        config.epochs = int(values.get("epochs", config.epochs))
        config.batch_size = int(values.get("batch_size", config.batch_size))
        config.learning_rate = float(values.get("learning_rate", config.learning_rate))
        config.seed = int(values.get("seed", config.seed))
        config.validation_ratio = float(
            values.get("validation_ratio", config.validation_ratio)
        )
        config.experiment_name = str(
            values.get("experiment_name", config.experiment_name)
        )
        config.registered_model_name = str(
            values.get("registered_model_name", config.registered_model_name)
        )
        config.model_output_path = str(
            values.get("model_output_path", config.model_output_path)
        )
        config.promotion_alias = str(promotion.get("alias", config.promotion_alias))
        config.minimum_accuracy = float(
            promotion.get("minimum_accuracy", config.minimum_accuracy)
        )
        config.maximum_accuracy_drop = float(
            promotion.get("maximum_accuracy_drop", config.maximum_accuracy_drop)
        )
        config.initialize_from_champion = bool(
            values.get("initialize_from_champion", config.initialize_from_champion)
        )
        config.initial_weights_path = values.get(
            "initial_weights_path", config.initial_weights_path
        )

    for argument, attribute in (
        ("data", "data"),
        ("additional_data", "additional_data"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("experiment_name", "experiment_name"),
        ("registered_model_name", "registered_model_name"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(config, attribute, value)
    if getattr(args, "fast_dev_run", False):
        config.fast_dev_run = True
        config.epochs = 1
        config.batch_size = min(config.batch_size, 8)
    if not 0 < config.validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    return config


def split_dataset(
    dataset: EyesDatasetLoader,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    indices = list(range(len(dataset)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=dataset.labels,
    )
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    train: bool,
) -> float:
    model.train(mode=train)
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            if optimizer is None:
                raise ValueError("optimizer is required for training")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device)).cpu().numpy().reshape(-1)
            y_pred.extend((outputs >= 0.5).astype(int).tolist())
            y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())
    return y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks([0, 1], labels=["closed", "opened"])
    axis.set_yticks([0, 1], labels=["closed", "opened"])
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, int(value), ha="center", va="center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def register_and_maybe_promote(
    *,
    model_uri: str,
    run_id: str,
    accuracy: float,
    config: TrainConfig,
) -> dict[str, Any]:
    from mlflow.tracking import MlflowClient

    result = mlflow.register_model(model_uri=model_uri, name=config.registered_model_name)
    version = str(result.version)
    client = MlflowClient()
    champion_accuracy: float | None = None
    champion_version: str | None = None
    try:
        champion = client.get_model_version_by_alias(
            config.registered_model_name,
            config.promotion_alias,
        )
        champion_version = str(champion.version)
        if champion.run_id:
            champion_accuracy = client.get_run(champion.run_id).data.metrics.get("accuracy")
    except Exception:  # noqa: BLE001 - first model has no alias
        pass

    threshold_ok = accuracy >= config.minimum_accuracy
    comparison_ok = (
        champion_accuracy is None
        or accuracy >= champion_accuracy - config.maximum_accuracy_drop
    )
    promoted = threshold_ok and comparison_ok
    if promoted:
        client.set_registered_model_alias(
            config.registered_model_name,
            config.promotion_alias,
            version,
        )
    client.set_model_version_tag(
        config.registered_model_name,
        version,
        "mlops.promotion_status",
        "promoted" if promoted else "rejected",
    )
    client.set_model_version_tag(
        config.registered_model_name,
        version,
        "mlops.validation_accuracy",
        str(accuracy),
    )
    mlflow.set_tags(
        {
            "mlops.registered_model_version": version,
            "mlops.promoted": str(promoted).lower(),
            "mlops.promotion_alias": config.promotion_alias,
        }
    )
    return {
        "version": version,
        "promoted": promoted,
        "alias": config.promotion_alias if promoted else None,
        "previous_champion_version": champion_version,
        "previous_champion_accuracy": champion_accuracy,
        "run_id": run_id,
    }


def initialize_model(config: TrainConfig, device: torch.device) -> tuple[nn.Module, str]:
    if config.initialize_from_champion:
        champion_uri = (
            f"models:/{config.registered_model_name}@{config.promotion_alias}"
        )
        try:
            model = mlflow.pytorch.load_model(champion_uri, map_location=device)
            return model.to(device), champion_uri
        except Exception as champion_error:  # noqa: BLE001 - explicit bootstrap fallback
            weights = os.getenv("MODEL_WEIGHTS_PATH") or config.initial_weights_path
            if not weights or not Path(weights).is_file():
                raise RuntimeError(
                    "Champion model is unavailable and bootstrap weights were not found"
                ) from champion_error
            model = MediumEyeCNN().to(device)
            model.load_state_dict(
                torch.load(weights, map_location=device, weights_only=True)
            )
            return model, f"weights:{weights}"
    return MediumEyeCNN().to(device), "random"


def train_model(config: TrainConfig) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((24, 24)), transforms.ToTensor()]
    )
    reference_dataset = EyesDatasetLoader(
        config.data,
        transform,
        max_per_class=16 if config.fast_dev_run else None,
    )
    reference_train, validation_dataset = split_dataset(
        reference_dataset,
        config.validation_ratio,
        config.seed,
    )
    train_parts: list[Dataset] = [reference_train]
    dataset_paths = [config.data]
    if config.additional_data and Path(config.additional_data).is_dir():
        try:
            additional_dataset = EyesDatasetLoader(
                config.additional_data,
                transform,
                max_per_class=16 if config.fast_dev_run else None,
            )
            train_parts.append(additional_dataset)
            dataset_paths.append(config.additional_data)
        except FileNotFoundError:
            pass
    train_dataset: Dataset = (
        train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, initialization_source = initialize_model(config, device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    output_dir = Path("artifacts/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(config.model_output_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.png"

    with mlflow.start_run(run_name="fast-dev-run" if config.fast_dev_run else None) as run:
        mlflow.log_params(
            {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "validation_ratio": config.validation_ratio,
                "dataset_paths": ",".join(dataset_paths),
                "reference_samples": len(reference_dataset),
                "training_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "initialization_source": initialization_source,
                "fast_dev_run": config.fast_dev_run,
            }
        )
        train_loss = 0.0
        validation_loss = 0.0
        for epoch in range(config.epochs):
            train_loss = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
            )
            validation_loss = run_epoch(
                model,
                validation_loader,
                criterion,
                None,
                device,
                train=False,
            )
            mlflow.log_metrics(
                {"train_loss": train_loss, "validation_loss": validation_loss},
                step=epoch,
            )
        y_true, y_pred = collect_predictions(model, validation_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        mlflow.log_metrics(metrics)
        torch.save(model.state_dict(), weights_path)
        save_confusion_matrix(y_true, y_pred, confusion_path)
        mlflow.log_artifact(str(weights_path), artifact_path="weights")
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        model_info = mlflow.pytorch.log_model(model, artifact_path="model")
        promotion = register_and_maybe_promote(
            model_uri=model_info.model_uri,
            run_id=run.info.run_id,
            accuracy=metrics["accuracy"],
            config=config,
        )
        summary = {
            "run_id": run.info.run_id,
            "experiment_name": config.experiment_name,
            "metrics": metrics,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "weights_path": str(weights_path),
            "registered_model_name": config.registered_model_name,
            "registered_model_version": promotion["version"],
            "promotion": promotion,
            "tracking_uri": tracking_uri,
            "initialization_source": initialization_source,
        }
        metrics_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Open Eyes Classifier with MLflow")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--additional-data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--registered-model-name", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_model(build_train_config(parse_args()))


if __name__ == "__main__":
    main()
