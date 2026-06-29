# Default image is the production FastAPI backend.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY open_eyes_classifier.py .
COPY backend ./backend
COPY scripts ./scripts
COPY configs ./configs
COPY params.yaml dvc.yaml eye_cnn_best_val_final.pth.dvc ./
COPY .dvc/config ./.dvc/config
COPY docker/dvc.config.local ./.dvc/config.local
COPY data/reference.dvc ./data/reference.dvc
COPY data/reference.dvc /dvc-pointers/reference.dvc

RUN mkdir -p data/reference data/incoming/opened data/incoming/closed \
    data/incoming/unlabeled reports/drift artifacts/models

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
