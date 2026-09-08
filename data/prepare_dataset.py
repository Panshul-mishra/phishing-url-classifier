# Builds a balanced subset from the PhiUSIIL Phishing URL Dataset (UCI).
# Only keeping url + label here - the dataset ships 50+ pre-computed
# columns but we're not using any of them since the whole point is to do
# our own feature extraction.
#
# dataset: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
#
# Dataset quirk found while testing (checked across the FULL raw file, not
# just a sample): every single legitimate URL in PhiUSIIL is a bare
# homepage - "https://www.levelup.com", never anything with a path. The
# phishing URLs, on the other hand, often do have a path/query
# (tracking links, fake login pages, etc). Train on that as-is and the
# model just learns "URL has a path -> phishing", which falls apart on
# literally any real link like wikipedia.org/wiki/Python or a github
# repo url. So we synthesize a batch of legit URLs WITH realistic paths
# (using real legit domains + made-up but plausible page paths) so "has a
# path" stops being a free shortcut for the model.
import random

import pandas as pd

RAW_PATH = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUT_PATH = "data/processed/urls_labeled.csv"
SAMPLE_PER_CLASS = 20000
PATH_AUGMENT_COUNT = 10000
RANDOM_STATE = 42

WORDS = [
    "about", "contact", "team", "careers", "pricing", "faq", "support",
    "help", "docs", "guide", "tutorial", "review", "update", "release",
    "announcement", "python", "javascript", "climate", "health", "space",
    "economy", "election", "startup", "design", "recipe", "travel",
    "fitness", "music", "movie", "book", "history", "science", "ai",
    "privacy", "terms", "signup", "login", "profile", "settings", "cart",
    "checkout", "order", "shipping", "returns", "warranty", "manual",
]

PATH_TEMPLATES = [
    "/{w}",
    "/{w}/{w2}",
    "/blog/{w}-{w2}",
    "/blog/2024/{w}-{w2}",
    "/news/{w}",
    "/article/{n}/{w}-{w2}",
    "/product/{n}",
    "/products/{w}",
    "/category/{w}/{w2}",
    "/docs/{w}/{w2}",
    "/wiki/{W}",
    "/questions/{n}/{w}-{w2}",
    "/user/{n}/profile",
    "/watch?v={n}",
    "/search?q={w}",
    "/{w}/{w2}/{n}",
]


def random_path(rng: random.Random) -> str:
    template = rng.choice(PATH_TEMPLATES)
    return template.format(
        w=rng.choice(WORDS),
        w2=rng.choice(WORDS),
        W=rng.choice(WORDS).capitalize(),
        n=rng.randint(1, 99999),
    )


def main():
    rng = random.Random(RANDOM_STATE)
    df = pd.read_csv(RAW_PATH, usecols=["URL", "Domain", "label"])
    df = df.rename(columns={"URL": "url", "label": "label"})
    df = df.dropna(subset=["url", "Domain"])
    df = df.drop_duplicates(subset="url")

    legit = df[df.label == 1]
    phish = df[df.label == 0]

    n = min(SAMPLE_PER_CLASS, len(legit), len(phish))
    legit_sample = legit.sample(n=n, random_state=RANDOM_STATE)
    phish_sample = phish.sample(n=n, random_state=RANDOM_STATE)

    # synthetic legit urls with a real path, built from real legit domains
    aug_domains = legit["Domain"].drop_duplicates().sample(
        n=PATH_AUGMENT_COUNT, random_state=RANDOM_STATE, replace=True
    ).tolist()
    path_urls = pd.DataFrame({
        "url": [f"https://{d}{random_path(rng)}" for d in aug_domains],
        "label": 1,
    })

    # extra real phishing urls to keep both classes the same size
    remaining_phish = phish[~phish.index.isin(phish_sample.index)]
    extra_phish = remaining_phish.sample(
        n=min(len(path_urls), len(remaining_phish)), random_state=RANDOM_STATE
    )[["url", "label"]]

    combined = pd.concat([
        legit_sample[["url", "label"]],
        phish_sample[["url", "label"]],
        path_urls,
        extra_phish,
    ])
    combined = combined.drop_duplicates(subset="url").sample(frac=1, random_state=RANDOM_STATE)

    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {OUT_PATH}")
    print(combined.label.value_counts())


if __name__ == "__main__":
    main()
