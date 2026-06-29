# DVC + MinIO

DVC tracks:

- `data/reference` through `data/reference.dvc`;
- `eye_cnn_best_val_final.pth` through its `.dvc` pointer.
- the reproducible baseline training stage and operational drift stage in
  `dvc.yaml`;
- the exact baseline dependency/output hashes in committed `dvc.lock`.

Training parameters are read from and tracked in `configs/train.yaml`; drift
thresholds are read from and tracked in `params.yaml`.
Baseline training deliberately sets `additional_dataset_path: null`. Production
samples from `data/incoming` are mutable and are consumed only by retraining.
The drift stage is marked `always_changed` because every run evaluates the
current production state rather than a versioned training snapshot.

The committed remote is `s3://mlops-eyes/dvc`. Its endpoint is local MinIO.
Credentials and endpoint overrides belong in environment variables and
`.dvc/config.local`, never in Git.

```bash
docker compose up -d minio createbucket
dvc remote modify --local minio endpointurl http://localhost:9000
dvc push data/reference.dvc eye_cnn_best_val_final.pth.dvc
dvc pull
dvc repro train
dvc status train
```

In Minikube an init container changes the endpoint to
`http://minio-service:9000`, copies the committed DVC pointer into the mounted
PVC, restores the data and copies bootstrap weights
to a shared `emptyDir`. The backend image also includes
`docker/dvc.config.local` with `core.no_scm=true`, because image builds
intentionally exclude Git metadata.

Each Kubernetes retrain Job performs its own DVC pull of bootstrap weights. This
keeps the fallback path available even when MLflow has no reachable champion.
