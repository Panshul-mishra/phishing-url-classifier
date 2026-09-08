# Trains the random forest on the features from features/extractor.py and
# dumps the model + feature order + metrics so the API can load them.
#
# run with:  venv/Scripts/python.exe -m model.train
import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features.extractor import FEATURE_NAMES, extract_features

DATA_PATH = "data/processed/urls_labeled.csv"
MODEL_PATH = "model/artifacts/random_forest.joblib"
FEATURE_NAMES_PATH = "model/artifacts/feature_names.json"
METRICS_PATH = "model/artifacts/metrics.json"
RANDOM_STATE = 42


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Extracting features for {len(df)} URLs...")
    start = time.time()
    rows = [extract_features(u) for u in df["url"]]
    print(f"  done in {time.time() - start:.1f}s")
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def main():
    df = pd.read_csv(DATA_PATH)
    # PhiUSIIL label convention: 1 = legitimate, 0 = phishing.
    # We flip it so our model's positive class (1) = phishing, which makes
    # "score" in the API response read naturally as "phishing probability".
    y = 1 - df["label"]

    X = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    print("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": dict(
            sorted(
                zip(FEATURE_NAMES, clf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importances"}, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

    joblib.dump(clf, MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved feature names -> {FEATURE_NAMES_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
