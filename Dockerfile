# Deploys to Render.com (free tier, no card needed). Render sets $PORT
# itself, so the CMD below falls back to 7860 only for local `docker run`.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features/ features/
COPY api/ api/
COPY dashboard/ dashboard/
COPY model/artifacts/random_forest.joblib model/artifacts/random_forest.joblib
COPY model/artifacts/feature_names.json model/artifacts/feature_names.json

EXPOSE 7860
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}
