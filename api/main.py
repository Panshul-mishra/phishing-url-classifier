# Client layer entry point + layer 3/4 glue.
#
# GET  /                 -> the demo dashboard
# GET  /api/v1/health    -> liveness check
# POST /api/v1/check     -> the actual pipeline: url in, score/level/why out
#
# run locally with: venv/Scripts/uvicorn.exe api.main:app --reload --port 8000
import os
import time

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.explain import build_explanation
from features.extractor import FEATURE_NAMES, extract_features

# resolved relative to this file, not the process cwd - uvicorn gets
# started from all kinds of working directories (the preview tool, a
# Dockerfile, HF Spaces) and this shouldn't break depending on that
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "artifacts", "random_forest.joblib")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

app = FastAPI(title="Phishing URL Classifier", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # a browser extension calls this from an extension origin, not a normal site
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = joblib.load(MODEL_PATH)
_explainer = shap.TreeExplainer(_model)


class CheckRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


class CheckResponse(BaseModel):
    url: str
    score: float
    level: str
    explanation: list[str]
    latency_ms: float


def score_to_level(score: float) -> str:
    if score < 0.2:
        return "Safe"
    if score < 0.4:
        return "Low Risk"
    if score < 0.6:
        return "Medium Risk"
    if score < 0.8:
        return "High Risk"
    return "Phishing"


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/check", response_model=CheckResponse)
def check_url(req: CheckRequest):
    start = time.time()

    features = extract_features(req.url)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    proba = _model.predict_proba(X)[0][1]

    shap_raw = _explainer.shap_values(X)
    # TreeExplainer on a binary RandomForestClassifier returns either a
    # (n_samples, n_features, n_classes) array or a 2-item list depending
    # on the shap version - handle both so this doesn't break on upgrade.
    if isinstance(shap_raw, list):
        contributions = shap_raw[1][0]
    else:
        contributions = shap_raw[0][:, 1] if shap_raw.ndim == 3 else shap_raw[0]
    shap_values = dict(zip(FEATURE_NAMES, contributions.tolist()))

    explanation = build_explanation(features, shap_values)
    if not explanation:
        explanation = ["no strong risk signals detected"] if proba < 0.5 else ["multiple minor risk signals combined"]

    return CheckResponse(
        url=req.url,
        score=round(float(proba), 4),
        level=score_to_level(proba),
        explanation=explanation,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
