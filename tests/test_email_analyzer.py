"""Unit tests for email phishing detection module."""

import os
import tempfile

import pytest

from src.email_analyzer import (
    analyze_email,
    analyze_content,
    analyze_sender,
    analyze_urls,
    analyze_attachments,
    calculate_risk_score,
    extract_urls_from_html,
    extract_urls_from_text,
    parse_email_content,
    parse_email_file,
)


class TestExtractUrls:
    def test_extract_urls_from_text(self):
        text = "Visit https://example.com for info"
        urls = extract_urls_from_text(text)
        assert "https://example.com" in urls

    def test_extract_urls_from_text_multiple(self):
        text = "Go to https://a.com and http://b.org/path"
        urls = extract_urls_from_text(text)
        assert "https://a.com" in urls
        assert "http://b.org/path" in urls

    def test_extract_urls_from_text_empty(self):
        assert extract_urls_from_text("") == []
        assert extract_urls_from_text(None) == []

    def test_extract_urls_from_html(self):
        html = '<a href="https://evil.com">Click here</a>'
        urls = extract_urls_from_html(html)
        assert len(urls) == 1
        assert urls[0]["url"] == "https://evil.com"
        assert urls[0]["visible_text"] == "Click here"

    def test_extract_urls_from_html_empty(self):
        assert extract_urls_from_html("") == []
        assert extract_urls_from_html(None) == []


class TestAnalyzeSender:
    def test_sender_reply_to_mismatch(self):
        inds = analyze_sender(
            "User <user@paypal.com>",
            "user@evil.com",
        )
        assert any(i["category"] == "sender_mismatch" for i in inds)

    def test_sender_match_no_issue(self):
        inds = analyze_sender(
            "User <user@paypal.com>",
            "user@paypal.com",
        )
        assert not any(i["category"] == "sender_mismatch" for i in inds)

    def test_sender_impersonation(self):
        inds = analyze_sender(
            "PayPal Security <security@paypal-verification.xyz>",
            "security@paypal-verification.xyz",
        )
        assert any(i["category"] == "sender_impersonation" for i in inds)

    def test_empty_sender(self):
        inds = analyze_sender("", "")
        assert inds == []


class TestAnalyzeContent:
    def test_credential_request(self):
        inds = analyze_content(
            "Urgent: Verify your password",
            "Please enter your password to verify your account",
            "",
        )
        assert any(i["category"] == "credential_request" for i in inds)

    def test_urgent_language(self):
        inds = analyze_content(
            "Account suspended!",
            "Your account will be closed immediately if you do not act now.",
            "",
        )
        assert any(i["category"] == "urgent_language" for i in inds)

    def test_financial_request(self):
        inds = analyze_content(
            "Invoice payment required",
            "Please provide your banking information for the wire transfer.",
            "",
        )
        assert any(i["category"] == "financial_request" for i in inds)

    def test_reward_claim(self):
        inds = analyze_content(
            "You won a prize!",
            "Congratulations, you have won a free gift card!",
            "",
        )
        assert any(i["category"] == "reward_claim" for i in inds)

    def test_clean_content(self):
        inds = analyze_content(
            "Meeting tomorrow",
            "Let's meet at the office to discuss the project timeline.",
            "",
        )
        assert inds == []


class TestAnalyzeAttachments:
    def test_risky_attachment(self):
        atts = [{"filename": "document.exe", "content_type": "application/octet-stream", "size": 1000}]
        inds = analyze_attachments(atts)
        assert any(i["category"] == "risky_attachment" for i in inds)

    def test_safe_attachment(self):
        atts = [{"filename": "document.pdf", "content_type": "application/pdf", "size": 1000}]
        inds = analyze_attachments(atts)
        assert not any(i["category"] == "risky_attachment" for i in inds)

    def test_no_attachments(self):
        inds = analyze_attachments([])
        assert inds == []


class TestCalculateRiskScore:
    def test_low_score(self):
        result = calculate_risk_score([], [], [], [])
        assert result["score"] == 0
        assert result["category"] == "low"

    def test_high_score(self):
        content = [{"category": "credential_request", "severity": "high",
                    "indicator": "Password request", "reason": "Asks for password"}]
        sender = [{"category": "sender_impersonation", "severity": "high",
                   "indicator": "Brand impersonation", "reason": "PayPal in display name"}]
        result = calculate_risk_score(content, sender, [], [])
        assert result["score"] >= 60
        assert result["category"] == "high"

    def test_score_capped_at_100(self):
        content = [{"category": "credential_request", "severity": "high",
                    "indicator": "Password request", "reason": "Asks for password"}] * 10
        result = calculate_risk_score(content, [], [], [])
        assert result["score"] == 100


class TestAnalyzeEmail:
    def test_legitimate_email(self):
        result = analyze_email(
            sender="newsletter@example.com",
            reply_to="newsletter@example.com",
            subject="Monthly newsletter",
            body_text="Here is our monthly newsletter with updates.",
            body_html="",
        )
        assert result["risk_category"] in ("low", "medium")
        assert "risk_score" in result

    def test_phishing_email_credentials(self):
        result = analyze_email(
            sender="PayPal Security <security@paypal-verification.xyz>",
            reply_to="security@evil.com",
            subject="Urgent: Verify your password",
            body_text="Your account is suspended. Please enter your password immediately to verify.",
            body_html="",
        )
        assert result["risk_category"] == "high"
        assert result["risk_score"] > 50

    def test_email_with_suspicious_urls(self):
        result = analyze_email(
            sender="user@example.com",
            reply_to="user@example.com",
            subject="Invoice",
            body_text="Please pay https://evil-phishing.xyz/pay now",
            body_html="",
        )
        assert len(result["urls"]) > 0

    def test_sender_reply_to_mismatch(self):
        result = analyze_email(
            sender="support@paypal.com",
            reply_to="support@evil.xyz",
            subject="Account update",
            body_text="Please update your account.",
            body_html="",
        )
        assert any(i["category"] == "sender_mismatch" for i in result["indicators"])

    def test_html_email_with_suspicious_link(self):
        result = analyze_email(
            sender="user@example.com",
            reply_to="user@example.com",
            subject="Click here",
            body_text="",
            body_html='<a href="https://phishing.bid">Verify account</a>',
        )
        assert len(result["urls"]) > 0
        assert result["urls"][0]["visible_text"] == "Verify account"

    def test_email_with_risky_attachment(self):
        result = analyze_email(
            sender="user@example.com",
            reply_to="user@example.com",
            subject="Document",
            body_text="Please see attached",
            body_html="",
            attachments=[{"filename": "invoice.exe", "content_type": "application/octet-stream", "size": 5000}],
        )
        assert any(i["category"] == "risky_attachment" for i in result["indicators"])

    def test_empty_email_raises(self):
        with pytest.raises(ValueError):
            analyze_email(sender="", reply_to="", subject="", body_text="", body_html="")


class TestParseEmailFile:
    def test_parse_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("From: user@example.com\nSubject: Test\n\nHello world\n")
            path = f.name
        try:
            parsed = parse_email_file(path)
            assert parsed["sender"] == "user@example.com"
            assert parsed["subject"] == "Test"
            assert "Hello" in parsed["body_text"]
        finally:
            os.unlink(path)

    def test_parse_nonexistent_file(self):
        with pytest.raises(ValueError):
            parse_email_file("/nonexistent/file.eml")

    def test_parse_unsupported_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("not an email")
            path = f.name
        try:
            with pytest.raises(ValueError):
                parse_email_file(path)
        finally:
            os.unlink(path)