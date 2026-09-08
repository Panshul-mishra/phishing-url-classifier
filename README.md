# Phishing URL Classifier

Panshul · BCA · Enrollment No. A1004824065

A phishing website classifier that scores a URL using a random forest running behind a small cloud-hosted API, built around the required 4-layer design (client -> feature extraction -> model inference -> response).

## What makes this version different

Most implementations of this assignment stop at: URL in, lexical features out (length, digit count, has-IP, etc.), random forest, 0/1 label out. This one adds two things on top of that:

1. **Brand-impersonation detection.** Alongside the usual lexical features, the feature extraction layer checks the domain against a list of ~90 commonly-spoofed brands (PayPal, banks, Microsoft, social platforms, etc.) using edit distance and a few substring/decoy checks. This catches things pure lexical features miss - `paypa1.com`, `paypal.com.verify-account.ru`, `epicgames-freebies.com` - domains that can look "clean" lexically (short, HTTPS, no weird TLD) while clearly trying to impersonate a real brand.
2. **Explained output, not just a label.** The response layer returns a risk *level* (Safe / Low / Medium / High / Phishing, not just phishing-or-not) plus a short plain-English explanation of what drove the score, generated from SHAP values at inference time.

## Architecture

```
Client Layer          browser extension / web dashboard / raw API call, sends a URL
      |
Feature Extraction    features/extractor.py -> 28 numeric features (22 lexical + 6 brand-similarity)
      |
Model Inference       random forest (scikit-learn), served via FastAPI, deployed to Hugging Face Spaces
      |
Response Layer        { score, level, explanation[] } as JSON
```

- `features/lexical.py` - standard URL features: length, digit ratio, hyphen count, entropy, suspicious TLD, IP-as-host, shortener detection, etc.
- `features/brand_similarity.py` - the differentiator. Tokenizes the domain and checks each token against the brand list for exact matches, homoglyph-normalized matches (`paypa1` -> `paypal`), and near-miss edit distance (`linkedln` vs `linkedin`), plus a check for a brand name stuffed into a decoy subdomain or the URL path.
- `features/extractor.py` - combines both into one ordered feature vector; this is the single place both training and the live API pull from, so they can't drift apart.
- `model/train.py` - trains the RandomForestClassifier and saves it + metrics.
- `model/ablation.py` - trains a lexical-only version of the model for comparison, to check the brand features are actually earning their place (see Evaluation below).
- `api/main.py` - FastAPI app: loads the model once, extracts features per request, runs SHAP, builds the response.
- `api/explain.py` - turns SHAP contributions into short human-readable reasons.
- `dashboard/` - the web client (also the FastAPI static root).
- `browser_extension/` - a Manifest V3 extension that checks the active tab's URL.

## Dataset

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI, 2024) - ~235k labeled URLs. Only the raw URL and label are used; the dataset's own 50+ pre-computed columns are ignored, since the point of this project is to do that feature engineering ourselves.

**A dataset problem I ran into and had to fix:** every single legitimate URL in this dataset is a bare homepage (`https://www.levelup.com`), while a good chunk of the phishing URLs have paths. Train on that directly and the model just learns "URL has a path = phishing" - which breaks on literally any normal link, e.g. `wikipedia.org/wiki/Python`. I caught this by testing the trained model on real-world URLs before calling it done, not just trusting the held-out accuracy number. Fixed it in `data/prepare_dataset.py` by synthesizing ~10k legitimate URLs with realistic paths (real legit domains + made-up but plausible page paths), so "has a path" stops being a free shortcut. Worth mentioning in a viva since it's the kind of dataset bias that's easy to miss if you only look at the accuracy score.

To rebuild the dataset:
```
venv/Scripts/python.exe data/prepare_dataset.py
```

## Setup

```
python -m venv venv
venv/Scripts/pip install -r requirements.txt
venv/Scripts/python.exe data/prepare_dataset.py
venv/Scripts/python.exe -m model.train
venv/Scripts/uvicorn.exe api.main:app --reload --port 8000
```
Then open `http://localhost:8000`.

## Evaluation

On a held-out 20% split (12,000 URLs): **99.0% accuracy**, 0.994 precision, 0.986 recall, 0.998 ROC-AUC (see `model/artifacts/metrics.json`).

That number alone doesn't say much though - a model can hit 99% on a random split just from surface cues like HTTPS presence. To actually test whether the brand-similarity features matter, `model/ablation.py` trains a second model using only the lexical features and compares both against `data/hard_examples.csv`, a small hand-written set of realistic brand-impersonation URLs (clean HTTPS, no suspicious TLD, no giveaway hyphens where possible) mixed with unrelated legitimate sites.

Result (see `model/artifacts/hard_set_comparison.json`): both models catch the obvious cases, but the full model assigns noticeably higher confidence to brand-impersonation attempts - e.g. `epicgames-freebies.com` scores 0.68 without brand features vs 0.81 with them, and during development the brand features fixed three concrete misses (`linkedln-security.com`, `discord-nitro-gift.com`, `epicgames-freebies.com`) that the lexical-only feature set had no way to catch, since these domains look structurally clean otherwise.

**Honest limitation:** a small number of concatenated brand+word domains with no other red flag at all (no digits, no hyphens, common TLD, e.g. `paypalsecurity.com`) still slip through even with brand features, because they look almost identical to a normal short-domain legitimate site from the URL alone. Fixing this properly would need something beyond URL text - domain age (WHOIS), certificate issue date, or actual page content.

## Deployment

Deployed as a Docker container to **Render** (free tier, no credit card) rather than AWS/GCP, since this is a class project with no budget. Live at:

**https://phishing-url-classifier-7fm7.onrender.com**

Steps: push this repo to GitHub, create a Render Web Service pointed at the repo, Render detects the `Dockerfile` and builds/deploys automatically. The free tier spins down after 15 minutes idle (~1 minute cold start on the next request) - if demoing live, open the link a minute before you need it.

The API also runs locally the same way via `uvicorn` (see Setup), and the same Docker image runs on any container host if a different cloud deploy is ever needed.

## API

```
POST /api/v1/check
{ "url": "https://paypa1.com/login" }

->

{
  "url": "https://paypa1.com/login",
  "score": 0.87,
  "level": "Phishing",
  "explanation": [
    "the domain text is very close to a well-known brand name",
    "the URL has an unusually high proportion of digits"
  ],
  "latency_ms": 4.2
}
```

## Browser extension

`browser_extension/` is a Manifest V3 extension - load it unpacked via `chrome://extensions` (Developer mode -> Load unpacked). It reads the active tab's URL and calls the deployed `/api/v1/check` endpoint on Render.
