# Feature extraction layer - turns a raw url into the numeric vector the
# model actually sees. Both training (model/train.py) and the live API
# import extract_features from here so they can't drift apart.
from features.brand_similarity import BRAND_FEATURE_NAMES, extract_brand_features
from features.lexical import LEXICAL_FEATURE_NAMES, extract_lexical_features

FEATURE_NAMES = LEXICAL_FEATURE_NAMES + BRAND_FEATURE_NAMES


def extract_features(url: str) -> dict:
    lexical = extract_lexical_features(url)
    brand = extract_brand_features(
        host=lexical["_host"],
        registrable_labels=lexical["_registrable_labels"],
        full_url=url,
        is_ip_host=bool(lexical["has_ip_host"]),
    )
    merged = {**lexical, **brand}
    # drop the private helper keys before returning the model-facing vector
    return {k: merged[k] for k in FEATURE_NAMES}


def extract_feature_vector(url: str) -> list:
    """Ordered list form, for feeding numpy/sklearn directly."""
    feats = extract_features(url)
    return [feats[name] for name in FEATURE_NAMES]
