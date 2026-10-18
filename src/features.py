"""Deterministic, offline-only URL feature extraction.

All features are computed from the URL string itself. No network requests
are made. The FeatureExtractor is scikit-learn compatible
(BaseEstimator + TransformerMixin) so it can be dropped into a Pipeline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from urllib.parse import urlsplit

import idna
import tldextract

# ---------------------------------------------------------------------------
# Feature name list (stable schema; bump config.feature_schema if changed)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "fragment_length",
    "num_dots_hostname",
    "num_hyphens_hostname",
    "num_underscores_hostname",
    "num_subdomains",
    "has_www",
    "has_port",
    "port_value",
    "is_ipv4_host",
    "is_ipv6_host",
    "is_idn_host",
    "uses_https",
    "has_at_symbol",
    "has_double_slash_in_path",
    "num_digits",
    "num_special_chars",
    "path_depth",
    "query_param_count",
    "tld_length",
    "is_common_tld",
    "hostname_entropy",
    "misleading_brand_in_subdomain",
    "suspicious_tld",
    "missing_scheme",
    "malformed",
]

NUM_FEATURES: int = len(FEATURE_NAMES)

# Tokens checked for misleading-brand subdomain heuristic
_BRAND_TOKEN_RE = re.compile(
    r"(paypal|microsoft|apple|amazon|google|facebook|netflix|chase|wellsfargo|bankofamerica)",
    re.IGNORECASE,
)

# IPv4 dotted-quad regex
_IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _is_ipv4(hostname: str) -> bool:
    return bool(_IPV4_RE.match(hostname or ""))


def _is_ipv6(hostname: str) -> bool:
    """Detect IPv6 hosts. urlsplit strips brackets, so handle both forms."""
    h = (hostname or "").strip()
    # Strip brackets if present (urlsplit may or may not include them)
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return ":" in h and bool(re.search(r"[0-9a-fA-F]", h))


def _extract_domain_parts(url: str) -> dict[str, str]:
    """Use tldextract to split a URL into subdomain, domain, suffix."""
    ext = tldextract.extract(url)
    # Use top_domain_under_public_suffix (replaces deprecated registered_domain)
    registered = getattr(ext, "top_domain_under_public_suffix", None) or getattr(
        ext, "registered_domain", ""
    ) or ""
    return {
        "subdomain": ext.subdomain or "",
        "domain": ext.domain or "",
        "suffix": ext.suffix or "",
        "registered_domain": registered,
    }


def extract_url_features(url: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract a deterministic feature dict from a single URL string."""
    cfg = config or {}
    misleading_brands = set(b.lower() for b in cfg.get("misleading_brands", []))
    suspicious_tlds = set(t.lower() for t in cfg.get("suspicious_tlds", []))
    common_tlds = set(t.lower() for t in cfg.get("common_tlds", []))
    max_url_length = cfg.get("max_url_length", 4096)

    result: dict[str, Any] = {name: 0 for name in FEATURE_NAMES}
    result["is_valid"] = False
    result["url_hash"] = ""
    result["raw_url"] = url

    if not isinstance(url, str) or url.strip() == "":
        result["malformed"] = 1
        return result

    if len(url) > max_url_length:
        result["malformed"] = 1
        return result

    if "\x00" in url:
        result["malformed"] = 1
        return result

    import hashlib

    result["url_hash"] = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    result["url_length"] = len(url)

    scheme_match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url)
    missing_scheme = not bool(scheme_match)
    result["missing_scheme"] = 1 if missing_scheme else 0

    parse_url = url if not missing_scheme else "http://" + url

    try:
        parts = urlsplit(parse_url)
    except Exception:
        result["malformed"] = 1
        return result

    hostname = parts.hostname or ""
    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""
    scheme = parts.scheme.lower() if parts.scheme else ""

    result["hostname_length"] = len(hostname)
    result["path_length"] = len(path)
    result["query_length"] = len(query)
    result["fragment_length"] = len(fragment)

    result["num_dots_hostname"] = hostname.count(".")
    result["num_hyphens_hostname"] = hostname.count("-")
    result["num_underscores_hostname"] = hostname.count("_")

    # Extract domain parts early for subdomain/TLD features
    domain_parts = _extract_domain_parts(parse_url)
    subdomain = domain_parts["subdomain"]

    # Subdomain count from tldextract
    result["num_subdomains"] = len([s for s in subdomain.split(".") if s]) if subdomain else 0

    # www subdomain presence
    result["has_www"] = 1 if subdomain.lower() == "www" else 0

    has_port = parts.port is not None
    result["has_port"] = 1 if has_port else 0
    result["port_value"] = parts.port if parts.port else 0

    result["is_ipv4_host"] = 1 if _is_ipv4(hostname) else 0
    result["is_ipv6_host"] = 1 if _is_ipv6(hostname) else 0

    is_idn = 0
    try:
        if hostname and any(ord(ch) > 127 for ch in hostname):
            idna.encode(hostname)
            is_idn = 1
    except Exception:
        pass
    result["is_idn_host"] = is_idn

    result["uses_https"] = 1 if scheme == "https" else 0

    # @ symbol indicates embedded credentials (user:pass@host)
    netloc = parts.netloc or ""
    result["has_at_symbol"] = 1 if "@" in netloc else 0

    after_authority = parse_url.split("://", 1)[-1] if "://" in parse_url else parse_url
    authority_end = len(hostname) + (len(str(parts.port)) if parts.port else 0)
    remainder = after_authority[authority_end:] if len(after_authority) > authority_end else ""
    result["has_double_slash_in_path"] = 1 if "//" in remainder else 0

    result["num_digits"] = sum(ch.isdigit() for ch in url)
    special = set("?&=#%+")
    result["num_special_chars"] = sum(ch in special for ch in url)

    path_segments = [seg for seg in path.split("/") if seg]
    result["path_depth"] = len(path_segments)

    result["query_param_count"] = len([p for p in query.split("&") if p]) if query else 0

    suffix = domain_parts["suffix"]
    result["tld_length"] = len(suffix)
    result["is_common_tld"] = 1 if suffix.lower() in common_tlds else 0
    result["suspicious_tld"] = 1 if suffix.lower() in suspicious_tlds else 0

    result["hostname_entropy"] = round(_shannon_entropy(hostname), 4)

    subdomain = domain_parts["subdomain"].lower()
    registered = domain_parts["registered_domain"].lower()
    misleading = 0
    if misleading_brands and subdomain:
        for brand in misleading_brands:
            if brand in subdomain and brand not in registered:
                misleading = 1
                break
    result["misleading_brand_in_subdomain"] = misleading

    result["is_valid"] = True
    result["malformed"] = 0
    return result


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """scikit-learn compatible transformer that converts URL strings to feature vectors."""

    def __init__(
        self,
        feature_names: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.feature_names = feature_names or FEATURE_NAMES
        self.config = config or {}
        self.n_features_in_ = len(self.feature_names)

    def fit(self, X: Any, y: Any = None) -> "FeatureExtractor":
        """No-op fit; the extractor is stateless."""
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Transform an iterable of URL strings into a 2D NumPy array."""
        if isinstance(X, pd.Series):
            urls = X.tolist()
        else:
            urls = list(X)

        rows: list[list[float]] = []
        for url in urls:
            feats = extract_url_features(url, config=self.config)
            row = [float(feats.get(name, 0.0)) for name in self.feature_names]
            rows.append(row)

        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(self.feature_names)