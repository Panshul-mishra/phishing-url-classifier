# Turns SHAP contributions into a short human-readable explanation for the
# response layer. Only fires for features that actually pushed the score
# in a meaningful direction - no point telling someone "url_length: 37"
# and calling it an explanation.

# feature name -> (phrase when it pushed toward phishing, phrase when it pushed toward legitimate)
PHRASES = {
    "is_https": ("the site is not served over HTTPS", "the site uses HTTPS"),
    "min_brand_edit_distance": ("the domain text is very close to a well-known brand name", None),
    "has_ip_host": ("the address is a raw IP rather than a domain name", None),
    "is_shortener": ("the link goes through a URL-shortening service", None),
    "suspicious_tld": ("the domain uses a top-level domain often abused for phishing", None),
    "brand_lookalike_flag": ("the domain is a near-miss spelling of a well-known brand", None),
    "brand_homoglyph_exact_match": ("the domain only reads as a known brand after swapping look-alike characters (e.g. 0 for o)", None),
    "brand_padded_with_brand_token": ("a well-known brand name is embedded in the domain next to other words", None),
    "brand_as_decoy_subdomain": ("a well-known brand name appears as a decoy subdomain while the real domain is unrelated", None),
    "brand_in_path_not_domain": ("a well-known brand name appears in the page path, not the actual domain", None),
    "num_hyphens": ("the domain has an unusually high number of hyphens", None),
    "domain_entropy": ("the domain looks like randomly generated characters", None),
    "digit_ratio": ("the URL has an unusually high proportion of digits", None),
    "num_subdomains": ("the URL has an unusually deep subdomain chain", None),
    "num_at_symbols": ("the URL contains an '@' symbol, often used to hide the real destination", None),
    "url_length": ("the URL is unusually long", None),
}

MIN_CONTRIBUTION = 0.01


def build_explanation(feature_values: dict, shap_values: dict, max_reasons: int = 3) -> list:
    ranked = sorted(shap_values.items(), key=lambda kv: -abs(kv[1]))
    reasons = []
    for name, contribution in ranked:
        if len(reasons) >= max_reasons or abs(contribution) < MIN_CONTRIBUTION:
            break
        if name not in PHRASES:
            continue
        toward_phishing_phrase, toward_legit_phrase = PHRASES[name]
        if contribution > 0 and toward_phishing_phrase:
            reasons.append(toward_phishing_phrase)
        elif contribution < 0 and toward_legit_phrase:
            reasons.append(toward_legit_phrase)
    return reasons
