# Brand impersonation / typosquat detection.
#
# Idea: a lot of phishing urls try to look like a real brand without
# actually being that brand's domain:
#   paypa1-secure-login.com          -> digit swapped in for a letter
#   paypal.com.verify-account.ru     -> real brand name stuffed in as a
#                                        fake subdomain, actual domain is
#                                        verify-account.ru
#   micros0ft-support.net            -> same idea, homoglyph swap
#
# none of the plain lexical stuff (length, digit count etc.) catches this,
# two urls can look equally "clean" there while one is obviously trying to
# be paypal. so we compare the domain against a list of brands that get
# spoofed a lot, using edit distance + a couple of substring checks.
import re

import Levenshtein

# Curated list of brands most frequently impersonated in phishing campaigns
# (banking, payments, tech/email, e-commerce, crypto, shipping, social).
KNOWN_BRANDS = [
    "paypal", "google", "amazon", "microsoft", "apple", "facebook",
    "instagram", "netflix", "chase", "bankofamerica", "wellsfargo",
    "americanexpress", "citibank", "hsbc", "barclays", "natwest",
    "santander", "irs", "dhl", "fedex", "ups", "usps", "linkedin",
    "twitter", "outlook", "office365", "dropbox", "adobe", "ebay",
    "walmart", "target", "coinbase", "binance", "blockchain", "steam",
    "steamcommunity", "roblox", "epicgames", "spotify", "whatsapp",
    "telegram", "icloud", "yahoo", "aol", "hotmail", "gmail", "sbi",
    "hdfcbank", "icicibank", "axisbank", "paytm", "phonepe", "googlepay",
    "venmo", "zelle", "westernunion", "moneygram", "wise", "revolut",
    "bankofindia", "rbi", "chase", "discover", "capitalone", "usbank",
    "twitch", "tiktok", "snapchat", "pinterest", "reddit", "github",
    "gitlab", "bitbucket", "salesforce", "docusign", "zoom", "slack",
    "att", "verizon", "tmobile", "comcast", "xfinity", "dhl", "aramex",
    "discord", "cashapp", "robinhood", "doordash", "uber", "lyft",
    "airbnb", "expedia", "booking", "hulu", "disneyplus", "playstation",
    "xbox", "nintendo", "indeed", "upwork", "fiverr",
]

# Plain "is this brand a substring of that string" checks below get used
# for decoy subdomains, path mentions, and no-separator padding. Short
# brand names are unsafe for that: "att" is inside "startups", "wise" is
# inside "otherwise", "ups" is inside half the dictionary. Restrict those
# specific checks to longer, more distinctive names - the token-level
# exact/edit-distance checks elsewhere don't have this problem since they
# compare a whole hyphen-delimited token, not an arbitrary substring.
SUBSTRING_SAFE_BRANDS = [b for b in KNOWN_BRANDS if len(b) >= 5]

# Common homoglyph / leetspeak substitutions used to dodge naive matching.
HOMOGLYPH_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
    "$": "s",
})


def _normalize(label: str) -> str:
    return label.lower().translate(HOMOGLYPH_MAP)


def _tokenize(label: str) -> list:
    """Split a domain label on non-letter characters so a compound name
    like "epicgames-freebies" or "linkedln-security" gets checked token
    by token, not just as one long string. Whole-label matching alone
    misses these: the extra "-freebies"/"-security" suffix inflates the
    edit distance against the brand past any sane threshold, even though
    the brand name (or a one-letter typo of it) is sitting right there."""
    tokens = [t for t in re.split(r"[^a-zA-Z]+", label.lower()) if len(t) >= 3]
    return tokens if tokens else [label.lower()]


def _closest_brand(label: str):
    """Smallest edit distance between any token of `label` and any known
    brand, using the homoglyph-normalized form of each token."""
    best_brand, best_dist = None, 999
    for token in _tokenize(label):
        norm = _normalize(token)
        for brand in KNOWN_BRANDS:
            dist = Levenshtein.distance(norm, brand)
            if dist < best_dist:
                best_brand, best_dist = brand, dist
    return best_brand, best_dist


_NEUTRAL_BRAND_FEATURES = {
    "min_brand_edit_distance": 99,
    "brand_lookalike_flag": 0,
    "brand_homoglyph_exact_match": 0,
    "brand_padded_with_brand_token": 0,
    "brand_as_decoy_subdomain": 0,
    "brand_in_path_not_domain": 0,
}


def extract_brand_features(
    host: str, registrable_labels: list, full_url: str, is_ip_host: bool = False
) -> dict:
    """
    host: e.g. "paypal.com.verify-account.ru"
    registrable_labels: domain_parts minus the TLD, e.g.
        ["paypal", "com", "verify-account"] for the host above
        (each of these can hide a decoy brand name)
    full_url: the raw URL, for decoy-in-path checks
    is_ip_host: brand-similarity is meaningless when the host is a raw IP -
        return neutral values so explanations don't cite a nonsense brand
        match (IP-address risk is already covered by has_ip_host).
    """
    if not host or is_ip_host:
        return dict(_NEUTRAL_BRAND_FEATURES)

    # The label most likely to be the "real" registrable name: the
    # left-most label of a typical www.brand.tld structure, or the whole
    # second-level domain if there's no subdomain.
    primary_label = registrable_labels[-1] if registrable_labels else host

    best_brand, best_dist = _closest_brand(primary_label)

    # A domain like "github.com" is just one token, and that token equals
    # a real brand name with NO character tricks - that's what a genuine
    # brand domain looks like, not a red flag on its own. A domain like
    # "paypa1.com" is also one token, but only matches "paypal" AFTER
    # undoing the digit-for-letter swap - that mismatch between the raw
    # text and what it visually reads as is the actual red flag.
    # A domain like "epicgames-freebies.com" is a different pattern again:
    # more than one token, and one of them happens to BE a real brand name
    # tacked onto extra words - a classic "brand + padding" phishing shape.
    primary_tokens_raw = [t.lower() for t in _tokenize(primary_label)]
    homoglyph_exact = 0
    padded_with_brand = 0
    if len(primary_tokens_raw) == 1:
        token = primary_tokens_raw[0]
        if token not in KNOWN_BRANDS and _normalize(token) in KNOWN_BRANDS:
            homoglyph_exact = 1
    else:
        for token in primary_tokens_raw:
            if token in KNOWN_BRANDS or _normalize(token) in KNOWN_BRANDS:
                padded_with_brand = 1
                break

    # Catches the case a hyphen-split miss: no separator at all, e.g.
    # "paypalsecurity.com" is one token to the tokenizer above, but a
    # brand name is still sitting right there as a substring.
    if not padded_with_brand:
        norm_label = _normalize(primary_label)
        for brand in SUBSTRING_SAFE_BRANDS:
            if brand in norm_label and brand != norm_label:
                padded_with_brand = 1
                break

    # Does a brand name appear anywhere in a NON-primary label? e.g.
    # "paypal" showing up as a subdomain/decoy rather than the actual
    # registered domain -> classic impersonation pattern.
    decoy_hit = 0
    other_labels = [l for l in registrable_labels if l != primary_label]
    for label in other_labels:
        norm = _normalize(label)
        for brand in SUBSTRING_SAFE_BRANDS:
            if brand in norm and brand != _normalize(primary_label):
                decoy_hit = 1
                break
        if decoy_hit:
            break

    # Lookalike: primary label is "close" to a brand (1-2 edit distance)
    # but not an exact match - the textbook typosquat signature.
    lookalike = int(0 < best_dist <= 2)

    # Brand name mentioned in the path/query but domain itself is unrelated,
    # e.g. totally-random-domain.com/paypal/login
    brand_in_path = 0
    path_and_query = full_url.split(host, 1)[-1].lower() if host in full_url else ""
    norm_path = _normalize(path_and_query)
    for brand in SUBSTRING_SAFE_BRANDS:
        if brand in norm_path and brand != _normalize(primary_label):
            brand_in_path = 1
            break

    return {
        "min_brand_edit_distance": best_dist,
        "brand_lookalike_flag": lookalike,
        "brand_homoglyph_exact_match": homoglyph_exact,
        "brand_padded_with_brand_token": padded_with_brand,
        "brand_as_decoy_subdomain": decoy_hit,
        "brand_in_path_not_domain": brand_in_path,
    }


BRAND_FEATURE_NAMES = [
    "min_brand_edit_distance",
    "brand_lookalike_flag",
    "brand_homoglyph_exact_match",
    "brand_padded_with_brand_token",
    "brand_as_decoy_subdomain",
    "brand_in_path_not_domain",
]
