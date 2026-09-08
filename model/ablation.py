# Ablation study: does the brand-similarity feature family actually help?
#
# Trains a second RF using only the plain lexical features (no brand stuff)
# on the same data, then evaluates both models on data/hard_examples.csv -
# a small hand-written set of realistic brand-impersonation phishing urls
# (https, clean-looking domains) mixed with unrelated legit sites. The
# random 80/20 split from the big dataset is not adversarial enough to
# show a difference, is_https alone gets you most of the way there, so
# this hard set is the actual test of whether brand-similarity earns its
# spot in the feature vector.
#
# run with: venv/Scripts/python.exe -m model.ablation
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from features.extractor import extract_features, FEATURE_NAMES
from features.lexical import LEXICAL_FEATURE_NAMES, extract_lexical_features

DATA_PATH = "data/processed/urls_labeled.csv"
HARD_SET_PATH = "data/hard_examples.csv"
LEXICAL_MODEL_PATH = "model/artifacts/random_forest_lexical_only.joblib"
FULL_MODEL_PATH = "model/artifacts/random_forest.joblib"
OUT_PATH = "model/artifacts/hard_set_comparison.json"
RANDOM_STATE = 42


def lexical_only_row(url: str) -> dict:
    feats = extract_lexical_features(url)
    return {k: feats[k] for k in LEXICAL_FEATURE_NAMES}


def train_lexical_only():
    df = pd.read_csv(DATA_PATH)
    y = 1 - df["label"]
    X = pd.DataFrame([lexical_only_row(u) for u in df["url"]], columns=LEXICAL_FEATURE_NAMES)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, LEXICAL_MODEL_PATH)
    return clf


def main():
    print("Training lexical-only baseline for comparison...")
    lexical_clf = train_lexical_only()
    full_clf = joblib.load(FULL_MODEL_PATH)

    hard = pd.read_csv(HARD_SET_PATH)
    y_true = hard["label"]

    X_lex = pd.DataFrame([lexical_only_row(u) for u in hard["url"]], columns=LEXICAL_FEATURE_NAMES)
    X_full = pd.DataFrame([extract_features(u) for u in hard["url"]], columns=FEATURE_NAMES)

    pred_lex = lexical_clf.predict(X_lex)
    pred_full = full_clf.predict(X_full)

    phishing_only = y_true == 1
    result = {
        "hard_set_size": len(hard),
        "lexical_only": {
            "accuracy": accuracy_score(y_true, pred_lex),
            "phishing_recall": recall_score(y_true, pred_lex),
        },
        "lexical_plus_brand": {
            "accuracy": accuracy_score(y_true, pred_full),
            "phishing_recall": recall_score(y_true, pred_full),
        },
        "missed_by_lexical_only": hard.loc[phishing_only & (pred_lex == 0), "url"].tolist(),
        "missed_by_full_model": hard.loc[phishing_only & (pred_full == 0), "url"].tolist(),
    }

    print(json.dumps(result, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved comparison -> {OUT_PATH}")


if __name__ == "__main__":
    main()
