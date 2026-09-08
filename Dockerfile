# Deploys to Hugging Face Spaces (Docker SDK) for free, no card needed.
# HF Spaces expects the app to listen on port 7860.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features/ features/
COPY api/ api/
COPY dashboard/ dashboard/
COPY model/artifacts/random_forest.joblib model/artifacts/random_forest.joblib
COPY model/artifacts/feature_names.json model/artifacts/feature_names.json

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
