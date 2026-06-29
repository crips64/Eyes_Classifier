# DVC + MinIO

DVC tracks:

- `data/reference` through `data/reference.dvc`;
- `eye_cnn_best_val_final.pth` through its `.dvc` pointer.
- the reproducible training/drift stages in `dvc.yaml`.

Training parameters are read from and tracked in `configs/train.yaml`; drift
thresholds are read from and tracked in `params.yaml`.

The committed remote is `s3://mlops-eyes/dvc`. Its endpoint is local MinIO.
Credentials and endpoint overrides belong in environment variables and
`.dvc/config.local`, never in Git.

```bash
docker compose up -d minio createbucket
dvc remote modify --local minio endpointurl http://localhost:9000
dvc push data/reference.dvc eye_cnn_best_val_final.pth.dvc
dvc pull
```

In Minikube an init container changes the endpoint to
`http://minio-service:9000`, copies the committed DVC pointer into the mounted
PVC, restores the data and copies bootstrap weights
to a shared `emptyDir`. The backend image also includes
`docker/dvc.config.local` with `core.no_scm=true`, because image builds
intentionally exclude Git metadata.
