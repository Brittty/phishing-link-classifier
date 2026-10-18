"""Unit tests for URL feature extraction."""

import pytest

from src.features import extract_url_features, FeatureExtractor, FEATURE_NAMES


class TestExtractUrlFeatures:
    def test_legitimate_url(self):
        feats = extract_url_features("https://www.google.com")
        assert feats["is_valid"] is True
        assert feats["uses_https"] == 1
        assert feats["malformed"] == 0
        assert feats["is_ipv4_host"] == 0
        assert feats["is_ipv6_host"] == 0

    def test_phishing_url_misleading_subdomain(self):
        feats = extract_url_features(
            "http://login-paypal.com.verify.xyz",
            config={"misleading_brands": ["paypal"], "suspicious_tlds": ["xyz"], "common_tlds": ["com"]},
        )
        assert feats["is_valid"] is True
        assert feats["uses_https"] == 0
        assert feats["suspicious_tld"] == 1

    def test_empty_url(self):
        feats = extract_url_features("")
        assert feats["is_valid"] is False
        assert feats["malformed"] == 1

    def test_none_url(self):
        feats = extract_url_features(None)
        assert feats["is_valid"] is False
        assert feats["malformed"] == 1

    def test_ipv4_host(self):
        feats = extract_url_features("http://192.168.1.1/login")
        assert feats["is_ipv4_host"] == 1

    def test_ipv6_host(self):
        feats = extract_url_features("http://[::1]/path")
        assert feats["is_ipv6_host"] == 1

    def test_port_in_url(self):
        feats = extract_url_features("https://example.com:8443/path")
        assert feats["has_port"] == 1
        assert feats["port_value"] == 8443

    def test_unicode_domain_idn(self):
        feats = extract_url_features("https://münchen.example.com")
        assert feats["is_valid"] is True

    def test_at_symbol_in_userinfo(self):
        feats = extract_url_features("http://user:pass@example.com")
        assert feats["has_at_symbol"] == 1

    def test_double_slash_not_counted_for_scheme(self):
        # https:// should NOT set has_double_slash_in_path
        feats = extract_url_features("https://example.com/path")
        assert feats["has_double_slash_in_path"] == 0

    def test_double_slash_in_path_is_counted(self):
        feats = extract_url_features("https://example.com//hidden//path")
        assert feats["has_double_slash_in_path"] == 1

    def test_query_string_params(self):
        feats = extract_url_features("https://example.com/a?b=1&c=2")
        assert feats["query_param_count"] == 2
        assert feats["query_length"] > 0

    def test_deterministic(self):
        a = extract_url_features("https://example.com/test?x=1")
        b = extract_url_features("https://example.com/test?x=1")
        assert a == b

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 29


class TestFeatureExtractor:
    def test_transform_returns_2d_array(self):
        extractor = FeatureExtractor()
        X = ["https://example.com", "http://evil.xyz"]
        result = extractor.fit_transform(X)
        assert result.shape == (2, len(FEATURE_NAMES))

    def test_transform_pandas_series(self):
        import pandas as pd
        extractor = FeatureExtractor()
        X = pd.Series(["https://example.com", "http://evil.xyz"])
        result = extractor.fit_transform(X)
        assert result.shape == (2, len(FEATURE_NAMES))

    def test_get_feature_names_out(self):
        extractor = FeatureExtractor()
        names = extractor.get_feature_names_out()
        assert names == FEATURE_NAMES

    def test_malformed_urls_do_not_crash(self):
        extractor = FeatureExtractor()
        X = ["", None, "not-a-url", "http://[::1]/ok"]
        result = extractor.fit_transform(X)
        assert result.shape == (4, len(FEATURE_NAMES))