# MLflow tracking and Model Registry

Training records parameters, losses, accuracy, precision, recall, F1, weights,
confusion matrix and a PyTorch model.

Every candidate is registered as `open-eyes-cnn`. Promotion to alias `champion`
requires:

- validation accuracy ≥ 0.85;
- accuracy no more than 0.01 below the current champion.

Retraining initializes from the current `champion` (with DVC bootstrap weights
as the explicit fallback). New labeled production samples are added only to the
training partition; promotion is evaluated on the deterministic reference
holdout so candidates remain comparable.

The API polls `models:/open-eyes-cnn@champion` every 30 seconds and swaps the
in-memory predictor only after the new model has loaded successfully.

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 \
python -m backend.src.train --config configs/train.yaml
```
