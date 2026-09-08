# Lexical/structural URL features. Only looks at the URL string itself,
# no page fetch - keeps this fast and means the client only ever sends
# a URL, never page content.
import math
import re
from collections import Counter
from urllib.parse import urlparse

IP_PATTERN = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$|^\[?[0-9a-fA-F:]+\]?$"
)

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "support",
    "click", "loan", "win", "review", "download", "racing", "party", "gq",
    "men", "date", "stream", "bid", "accountant", "cricket", "science",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy",
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _looks_like_ip(host: str) -> bool:
    host = host.strip("[]")
    return bool(IP_PATTERN.match(host))


def extract_lexical_features(url: str) -> dict:
    url = url.strip()
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url

    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    domain_parts = host.split(".") if host else []
    tld = domain_parts[-1].lower() if len(domain_parts) > 1 else ""
    registrable_labels = domain_parts[:-1] if len(domain_parts) > 1 else domain_parts
    num_subdomains = max(len(domain_parts) - 2, 0)

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    specials = len(url) - digits - letters

    tokens = [t for t in re.split(r"[^a-zA-Z0-9]", url) if t]
    longest_token = max((len(t) for t in tokens), default=0)

    return {
        "url_length": len(url),
        "domain_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_at_symbols": url.count("@"),
        "num_query_params": query.count("=") if query else 0,
        "num_digits": digits,
        "num_special_chars": specials,
        "digit_ratio": digits / len(url) if url else 0.0,
        "letter_ratio": letters / len(url) if url else 0.0,
        "num_subdomains": num_subdomains,
        "is_https": int(parsed.scheme == "https"),
        "has_ip_host": int(_looks_like_ip(host)) if host else 0,
        "has_port": int(parsed.port is not None),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(host.lower() in SHORTENER_DOMAINS),
        "domain_entropy": _shannon_entropy(host),
        "longest_token_length": longest_token,
        "num_tokens": len(tokens),
        # exposed for reuse by other feature modules, not fed to the model
        "_host": host,
        "_registrable_labels": registrable_labels,
    }


LEXICAL_FEATURE_NAMES = [
    k for k in extract_lexical_features("http://example.com").keys()
    if not k.startswith("_")
]
