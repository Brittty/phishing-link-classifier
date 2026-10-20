"""Email Phishing Detection Module.

Analyzes email content for phishing indicators using explainable rules.
Reuses the existing URL classifier for URL analysis within emails.

No URLs are visited, downloaded, or executed. No attachments are opened
or extracted. All analysis is local and offline.
"""

from __future__ import annotations

import email
import html as html_module
import re
from email.message import Message
from email.policy import default
from pathlib import Path
from typing import Any

from .config import Config
from .predict import predict_url
from .security import is_trusted_artifact, validate_url_input


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_EMAIL_SIZE = 5_000_000  # 5 MB
DEFAULT_MAX_ATTACHMENTS = 10
DEFAULT_SUSPICIOUS_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".com", ".scr", ".pif", ".vbs", ".vbe",
    ".js", ".jse", ".wsf", ".wsh", ".ps1", ".psm1", ".hta",
    ".msi", ".msp", ".cpl", ".jar", ".ws", ".wsf",
    ".docm", ".dotm", ".xlsm", ".xltm", ".pptm", ".potm", ".ppam",
    ".accdb", ".accde", ".accdr", ".accdt", ".mdb", ".mde",
    ".vb", ".vba", ".bas", ".cls", ".frm",
}

DEFAULT_SUSPICIOUS_KEYWORDS = [
    "urgent", "immediately", "suspended", "closure", "close your account",
    "account will be closed", "verify your account", "account verification",
    "security alert", "unauthorized access", "suspicious activity",
    "password", "otp", "one-time", "credential", "login", "sign in",
    "confirm your identity", "update your payment", "banking information",
    "credit card", "social security", "ssn", "pin", "pin number",
    "payment", "invoice", "overdue", "outstanding balance",
    "wire transfer", "gift card", "bitcoin", "crypto", "refund",
    "you have won", "congratulations", "prize", "reward", "lottery",
    "claim your", "free money", "inherit", "beneficiary",
    "account recovery", "help desk", "support team", "verify now",
    "click here", "act now", "limited time",
]


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_URL_REGEX = re.compile(
    r"https?://[^\s<>\"]+|www\.[^\s<>\"]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)

_URL_SHORTENERS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly",
    "adf.ly", "bit.do", "rebrand.ly", "cutt.ly", "shorte.st",
    "tiny.cc", "is.gd", "clk.de", "po.st", "bc.vc",
}


def extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from plain text or HTML-stripped text."""
    if not text:
        return []
    urls = _URL_REGEX.findall(text)
    cleaned: list[str] = []
    for u in urls:
        u = u.rstrip(".,;:!?)")
        if u not in cleaned:
            cleaned.append(u)
    return cleaned


def extract_urls_from_html(html_content: str) -> list[dict[str, str]]:
    """Extract URLs and visible link text from HTML content.

    Returns a list of dicts with keys: url, visible_text.
    """
    results: list[dict[str, str]] = []
    if not html_content:
        return results

    anchor_re = re.compile(
        r'<a\s+[^>]*href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in anchor_re.finditer(html_content):
        url = match.group(1).strip()
        visible = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        visible = html_module.unescape(visible)
        results.append({"url": url, "visible_text": visible})

    standalone = extract_urls_from_text(html_content)
    seen_urls = {r["url"] for r in results}
    for url in standalone:
        if url not in seen_urls:
            results.append({"url": url, "visible_text": ""})
            seen_urls.add(url)

    return results


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------


def parse_email_file(file_path: str | Path) -> dict[str, Any]:
    """Parse a .eml or .txt email file safely.

    Returns a dict with keys:
        sender, reply_to, subject, body_text, body_html,
        attachments (list of dicts with filename, content_type, size),
        raw_headers (dict)
    """
    path = Path(file_path).resolve()

    # Security: prevent path traversal
    if not path.is_file():
        raise ValueError(f"File not found: {file_path}")

    # Security: file size limit
    max_size = DEFAULT_MAX_EMAIL_SIZE
    if path.stat().st_size > max_size:
        raise ValueError(
            f"File exceeds maximum size of {max_size / 1_000_000:.1f} MB"
        )

    # Security: only allow .eml and .txt extensions
    if path.suffix.lower() not in {".eml", ".txt"}:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Only .eml and .txt are allowed."
        )

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        raise ValueError(f"Failed to read email file: {exc}") from exc

    return parse_email_content(content)


def parse_email_content(content: str) -> dict[str, Any]:
    """Parse raw email string content.

    Uses Python's standard email package. No network access is performed.
    """
    if not content or not content.strip():
        raise ValueError("Email content is empty")

    try:
        msg = email.message_from_string(content, policy=default)
    except Exception as exc:
        raise ValueError(f"Failed to parse email: {exc}") from exc

    # Extract headers
    sender = msg.get("From", "")
    reply_to = msg.get("Reply-To", "")
    subject = msg.get("Subject", "")

    # Decode headers if needed
    try:
        from email.header import decode_header, make_header

        sender = str(make_header(decode_header(sender))) if sender else ""
        reply_to = str(make_header(decode_header(reply_to))) if reply_to else ""
        subject = str(make_header(decode_header(subject))) if subject else ""
    except Exception:
        pass

    # Extract body
    body_text = ""
    body_html = ""
    attachments: list[dict[str, Any]] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get("Content-Disposition", "") or ""
            filename = part.get_filename()

            if filename:
                payload = part.get_payload(decode=True) or b""
                attachments.append(
                    {
                        "filename": filename,
                        "content_type": content_type,
                        "size": len(payload),
                        "disposition": disposition,
                    }
                )
            elif content_type == "text/plain" and "attachment" not in disposition:
                body_text = part.get_content() or ""
            elif content_type == "text/html" and "attachment" not in disposition:
                body_html = part.get_content() or ""
    else:
        content_type = msg.get_content_type()
        try:
            payload = msg.get_content()
        except Exception:
            try:
                raw = msg.get_payload(decode=True)
                payload = raw.decode("utf-8", errors="replace") if raw else ""
            except Exception:
                payload = ""
        if content_type == "text/html":
            body_html = payload or ""
        else:
            # Plain-text fallback: treat any non-multipart content as body text.
            # This handles .txt uploads and simple single-part .eml files.
            if isinstance(payload, str) and payload.strip():
                body_text = payload
            else:
                try:
                    raw = msg.get_payload()
                    if isinstance(raw, str) and raw.strip():
                        body_text = raw
                except Exception:
                    pass
        # Plain .txt uploads without MIME headers: message body may sit in
        # the payload even when content-type defaults to text/plain.
        if not body_text and not body_html:
            try:
                raw = msg.get_payload()
                if isinstance(raw, str) and raw.strip():
                    body_text = raw
            except Exception:
                pass

    return {
        "sender": sender,
        "reply_to": reply_to,
        "subject": subject,
        "body_text": body_text or "",
        "body_html": body_html or "",
        "attachments": attachments,
        "raw_headers": dict(msg.items()),
    }


# ---------------------------------------------------------------------------
# Sender / domain analysis
# ---------------------------------------------------------------------------


def _extract_domain(address: str) -> str:
    """Extract domain from an email address, handling 'Name <email>' format."""
    if not address or "@" not in address:
        return ""
    # Handle 'Display Name <email@domain>' format: extract bracketed email first
    bracket = re.search(r"<([^<>@\s]+@[^<>\s]+)>", address)
    email_part = bracket.group(1) if bracket else address.strip()
    # Strip any trailing '>' or whitespace leftovers
    email_part = email_part.strip().rstrip(">").strip()
    if "@" not in email_part:
        return ""
    return email_part.split("@")[-1].strip().lower().rstrip(">").strip()


def _extract_display_name(address: str) -> str:
    """Extract display name from 'Name <email@domain>' format."""
    if not address:
        return ""
    match = re.match(r'^["\']?([^"<>\']+)["\']?\s*<', address)
    if match:
        return match.group(1).strip()
    if "<" not in address and "@" in address:
        return ""
    return address.strip()


def analyze_sender(sender: str, reply_to: str) -> list[dict[str, Any]]:
    """Analyze sender and reply-to addresses for phishing indicators."""
    indicators: list[dict[str, Any]] = []

    sender_domain = _extract_domain(sender)
    reply_domain = _extract_domain(reply_to)
    sender_display = _extract_display_name(sender)

    # Reply-To mismatch
    if sender_domain and reply_domain and sender_domain != reply_domain:
        indicators.append(
            {
                "category": "sender_mismatch",
                "severity": "high",
                "indicator": "Sender and Reply-To domain mismatch",
                "reason": (
                    f"Sender domain '{sender_domain}' differs from Reply-To "
                    f"domain '{reply_domain}'. This is a common spoofing tactic."
                ),
                "sender_domain": sender_domain,
                "reply_to_domain": reply_domain,
            }
        )

    # Display name inconsistency: a well-known brand in the display name or
    # local part combined with a non-official sender domain.
    if sender_display and sender_domain:
        known_brands = ["paypal", "microsoft", "apple", "amazon", "google",
                        "facebook", "netflix", "chase", "wellsfargo", "bankofamerica"]
        display_lower = sender_display.lower()
        # Official domains that legitimately belong to each brand.
        official_domains = {
            "paypal": ("paypal.com",),
            "microsoft": ("microsoft.com", "outlook.com", "hotmail.com", "live.com"),
            "apple": ("apple.com", "icloud.com"),
            "amazon": ("amazon.com",),
            "google": ("google.com", "gmail.com"),
            "facebook": ("facebook.com", "meta.com"),
            "netflix": ("netflix.com",),
            "chase": ("chase.com", "jpmorganchase.com"),
            "wellsfargo": ("wellsfargo.com",),
            "bankofamerica": ("bankofamerica.com",),
        }
        local_part = sender_domain  # fallback
        try:
            bracket = re.search(r"<([^<>@\s]+@[^<>\s]+)>", sender)
            email_part = bracket.group(1) if bracket else sender.strip()
            local_part = email_part.split("@")[0].lower() if "@" in email_part else ""
        except Exception:
            pass
        for brand in known_brands:
            brand_in_name = brand in display_lower or brand in local_part
            if not brand_in_name:
                continue
            allowed = official_domains.get(brand, (f"{brand}.com",))
            if sender_domain not in allowed and not any(
                sender_domain == d or sender_domain.endswith("." + d) for d in allowed
            ):
                indicators.append(
                    {
                        "category": "sender_impersonation",
                        "severity": "high",
                        "indicator": f"Possible brand impersonation in sender",
                        "reason": (
                            f"Sender display name '{sender_display}' references "
                            f"'{brand}' but domain '{sender_domain}' does not match."
                        ),
                        "brand": brand,
                        "sender_display": sender_display,
                        "sender_domain": sender_domain,
                    }
                )
                break

    # Suspicious sender patterns
    if sender:
        sender_lower = sender.lower()
        suspicious_patterns = [
            r"noreply@.*\.(xyz|top|bid|win|club|online|site|review|country)",
            r"security@.*\.(xyz|top|bid|win)",
            r"account@.*\.(xyz|top|bid|win)",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, sender_lower):
                indicators.append(
                    {
                        "category": "suspicious_sender",
                        "severity": "medium",
                        "indicator": "Suspicious sender address pattern",
                        "reason": (
                            f"Sender address '{sender}' uses a suspicious "
                            f"pattern commonly seen in phishing campaigns."
                        ),
                        "sender": sender,
                    }
                )
                break

    return indicators


# ---------------------------------------------------------------------------
# Content analysis
# ---------------------------------------------------------------------------


def _count_keyword_matches(text: str, keywords: list[str]) -> list[str]:
    """Find which keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    return found


def analyze_content(
    subject: str,
    body_text: str,
    body_html: str,
    keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Analyze email subject and body for suspicious language."""
    indicators: list[dict[str, Any]] = []
    keywords = keywords or DEFAULT_SUSPICIOUS_KEYWORDS

    combined_text = f"{subject}\n{body_text}\n{body_html}"
    found_keywords = _count_keyword_matches(combined_text, keywords)

    if not found_keywords:
        return indicators

    urgency_kws = ["urgent", "immediately", "suspended", "closure",
                   "account will be closed", "act now", "limited time",
                   "verify now", "security alert", "suspicious activity"]
    credential_kws = ["password", "otp", "one-time", "credential", "login",
                      "sign in", "confirm your identity", "pin", "pin number"]
    financial_kws = ["banking information", "credit card", "social security",
                     "ssn", "payment", "invoice", "wire transfer", "gift card"]
    reward_kws = ["you have won", "congratulations", "prize", "reward",
                  "lottery", "claim your", "free money", "inherit"]

    categories: dict[str, list[str]] = {
        "urgent_language": [],
        "credential_request": [],
        "financial_request": [],
        "reward_claim": [],
    }

    for kw in found_keywords:
        kl = kw.lower()
        if any(u in kl for u in urgency_kws):
            categories["urgent_language"].append(kw)
        elif any(c in kl for c in credential_kws):
            categories["credential_request"].append(kw)
        elif any(f in kl for f in financial_kws):
            categories["financial_request"].append(kw)
        elif any(r in kl for r in reward_kws):
            categories["reward_claim"].append(kw)

    category_descriptions = {
        "urgent_language": ("Urgent or threatening language",
                            "Uses pressure tactics to urge immediate action."),
        "credential_request": ("Credential or password request",
                               "Asks for passwords, OTPs, or login credentials."),
        "financial_request": ("Financial information request",
                              "Requests banking, payment, or financial details."),
        "reward_claim": ("Unexpected reward or prize claim",
                         "Claims unexpected winnings, rewards, or inheritance."),
    }

    for cat, kws in categories.items():
        if kws:
            desc, reason = category_descriptions[cat]
            indicators.append(
                {
                    "category": cat,
                    "severity": "high" if cat in ("credential_request", "financial_request") else "medium",
                    "indicator": f"{desc}: {', '.join(sorted(set(kws)))}",
                    "reason": reason,
                    "matched_keywords": sorted(set(kws)),
                }
            )

    # Any leftover keywords that did not fit the four buckets -> generic bucket
    assigned = {k for v in categories.values() for k in v}
    leftover = [k for k in found_keywords if k not in assigned]
    if leftover:
        indicators.append(
            {
                "category": "suspicious_language",
                "severity": "medium",
                "indicator": f"Suspicious language: {', '.join(sorted(set(leftover)))}",
                "reason": "Contains language commonly seen in phishing emails.",
                "matched_keywords": sorted(set(leftover)),
            }
        )

    return indicators


# ---------------------------------------------------------------------------
# URL analysis (reuses existing URL classifier)
# ---------------------------------------------------------------------------


def analyze_urls(
    urls: list[dict[str, str]],
    model_path: str | Path | None = None,
    config: Config | None = None,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Analyze URLs extracted from email using the existing URL classifier.

    If no model is available or it fails trust checks, URLs are reported
    with rule-based indicators only (no ML prediction).
    """
    cfg = config or Config.load()
    results: list[dict[str, Any]] = []

    # Try to load the URL model
    pipeline = None
    model_available = False
    if model_path:
        if is_trusted_artifact(model_path, trusted_dir=cfg.trusted_model_dir):
            try:
                from .model import load_pipeline
                pipeline = load_pipeline(model_path)
                model_available = True
            except Exception:
                pipeline = None
                model_available = False

    for item in urls:
        url = item["url"]
        visible_text = item.get("visible_text", "")

        url_result: dict[str, Any] = {
            "url": url,
            "visible_text": visible_text,
            "rule_indicators": [],
            "ml_prediction": None,
            "ml_probability": None,
            "ml_available": False,
        }

        # Rule-based URL checks
        # 1. Unusually long URL
        if len(url) > 100:
            url_result["rule_indicators"].append(
                "Unusually long URL (may be obfuscated)"
            )

        # 2. IP address as host
        from .features import _is_ipv4, _is_ipv6
        from urllib.parse import urlsplit
        try:
            parts = urlsplit(url if "://" in url else "http://" + url)
            hostname = parts.hostname or ""
            if _is_ipv4(hostname) or _is_ipv6(hostname):
                url_result["rule_indicators"].append(
                    "Uses IP address as host (suspicious)"
                )
        except Exception:
            pass

        # 3. URL shortener
        try:
            parts = urlsplit(url if "://" in url else "http://" + url)
            hostname = parts.hostname or ""
        except Exception:
            hostname = ""
        if hostname in _URL_SHORTENERS:
            url_result["rule_indicators"].append(
                "Uses URL shortening service (destination hidden)"
            )

        # 4. Mismatched visible text vs URL
        if visible_text and visible_text.startswith("http"):
            if visible_text.lower() != url.lower():
                url_result["rule_indicators"].append(
                    "Visible link text differs from actual URL"
                )

        # 5. ML classification via the existing URL pipeline (never fetch the URL)
        if pipeline is not None:
            try:
                from .predict import predict_url as _predict_url

                pred = _predict_url(pipeline, url, threshold=threshold)
                if pred.get("is_valid"):
                    url_result["ml_available"] = True
                    url_result["ml_prediction"] = pred.get("prediction")
                    url_result["ml_probability"] = pred.get("probability")
                    url_result["ml_threshold"] = pred.get("threshold", threshold)
                # Invalid URL inputs simply keep ml_available=False
            except Exception:
                url_result["ml_available"] = False

        results.append(url_result)

    return results



# ---------------------------------------------------------------------------
# Attachment analysis
# ---------------------------------------------------------------------------


def analyze_attachments(
    attachments: list[dict[str, Any]],
    suspicious_extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Analyze attachment metadata for risky indicators.

    Never opens or extracts attachments. Only inspects filenames.
    """
    indicators: list[dict[str, Any]] = []
    exts = suspicious_extensions or DEFAULT_SUSPICIOUS_EXTENSIONS

    if len(attachments) > DEFAULT_MAX_ATTACHMENTS:
        indicators.append(
            {
                "category": "too_many_attachments",
                "severity": "medium",
                "indicator": "Excessive number of attachments",
                "reason": f"Email contains {len(attachments)} attachments (limit: {DEFAULT_MAX_ATTACHMENTS}).",
            }
        )

    for att in attachments:
        filename = att.get("filename", "")
        if not filename:
            continue

        ext = Path(filename).suffix.lower()
        if ext in exts:
            indicators.append(
                {
                    "category": "risky_attachment",
                    "severity": "high",
                    "indicator": f"Risky attachment: {filename}",
                    "reason": (
                        f"Attachment '{filename}' has extension '{ext}' "
                        f"which is commonly used for malware or macro-based attacks."
                    ),
                    "filename": filename,
                    "extension": ext,
                }
            )

    return indicators


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

SCORING_RULES = {
    "urgent_language": 15,
    "credential_request": 30,
    "financial_request": 30,
    "reward_claim": 15,
    "suspicious_language": 10,
    "sender_mismatch": 20,
    "sender_impersonation": 30,
    "suspicious_sender": 10,
    "url_long": 5,
    "url_ip_host": 10,
    "url_shortener": 8,
    "url_visible_mismatch": 10,
    "url_rule": 5,
    "url_ml_phishing": 15,
    "url_ml": 15,
    "risky_attachment": 15,
    "too_many_attachments": 5,
}

RISK_THRESHOLDS = {
    "low": 30,
    "medium": 60,
}


def calculate_risk_score(
    content_indicators: list[dict[str, Any]],
    sender_indicators: list[dict[str, Any]],
    url_results: list[dict[str, Any]],
    attachment_indicators: list[dict[str, Any]],
    scoring_rules: dict[str, int] | None = None,
    thresholds: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Calculate an explainable risk score from 0 to 100.

    Returns a dict with score, category, and breakdown.
    """
    rules = scoring_rules or SCORING_RULES
    thresh = thresholds or RISK_THRESHOLDS

    score = 0
    breakdown: list[dict[str, Any]] = []

    # Content indicators
    for ind in content_indicators:
        cat = ind["category"]
        points = rules.get(cat, 10)
        score += points
        breakdown.append(
            {
                "source": "content",
                "category": cat,
                "indicator": ind["indicator"],
                "reason": ind["reason"],
                "severity": ind["severity"],
                "points": points,
            }
        )

    # Sender indicators
    for ind in sender_indicators:
        cat = ind["category"]
        points = rules.get(cat, 10)
        score += points
        breakdown.append(
            {
                "source": "sender",
                "category": cat,
                "indicator": ind["indicator"],
                "reason": ind["reason"],
                "severity": ind["severity"],
                "points": points,
            }
        )

    # URL indicators
    for url_result in url_results:
        for ri in url_result.get("rule_indicators", []):
            if "long" in ri.lower():
                cat = "url_long"
            elif "ip address" in ri.lower():
                cat = "url_ip_host"
            elif "shortener" in ri.lower():
                cat = "url_shortener"
            elif "visible link text differs" in ri.lower():
                cat = "url_visible_mismatch"
            else:
                cat = "url_long"
            points = rules.get(cat, 5)
            score += points
            breakdown.append(
                {
                    "source": "url",
                    "category": cat,
                    "indicator": ri,
                    "reason": f"URL: {url_result['url']}",
                    "severity": "medium",
                    "points": points,
                    "url": url_result["url"],
                }
            )

        # ML URL classification
        if url_result.get("ml_available") and url_result.get("ml_prediction") == "phishing":
            points = rules.get("url_ml_phishing", 15)
            score += points
            breakdown.append(
                {
                    "source": "url_ml",
                    "category": "url_ml_phishing",
                    "indicator": "URL classified as phishing by ML model",
                    "reason": (
                        f"URL '{url_result['url']}' classified as phishing "
                        f"(probability: {url_result.get('ml_probability', 0):.2f})"
                    ),
                    "severity": "high",
                    "points": points,
                    "url": url_result["url"],
                }
            )

    # Attachment indicators
    for ind in attachment_indicators:
        cat = ind["category"]
        points = rules.get(cat, 10)
        score += points
        breakdown.append(
            {
                "source": "attachment",
                "category": cat,
                "indicator": ind["indicator"],
                "reason": ind["reason"],
                "severity": ind["severity"],
                "points": points,
            }
        )

    # Cap score at 100
    score = min(score, 100)

    # Determine category
    if score >= thresh["medium"]:
        category = "high"
    elif score >= thresh["low"]:
        category = "medium"
    else:
        category = "low"

    return {
        "score": score,
        "category": category,
        "breakdown": breakdown,
        "total_indicators": len(breakdown),
    }


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_email(
    sender: str = "",
    reply_to: str = "",
    subject: str = "",
    body_text: str = "",
    body_html: str = "",
    attachments: list[dict[str, Any]] | None = None,
    urls_from_body: list[dict[str, str]] | None = None,
    model_path: str | Path | None = None,
    config: Config | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Complete email analysis pipeline.

    Returns a dict with:
        risk_score, risk_category, indicators, urls, attachments,
        summary, safe_actions
    """
    if not any([(sender or "").strip(), (reply_to or "").strip(),
                (subject or "").strip(), (body_text or "").strip(),
                (body_html or "").strip()]) and not attachments and not urls_from_body:
        raise ValueError("Email content is empty: no sender, subject, or body provided")
    cfg = config or Config.load()
    attachments = attachments or []

    # 1. Sender analysis
    sender_indicators = analyze_sender(sender, reply_to)

    # 2. Content analysis
    content_indicators = analyze_content(subject, body_text, body_html)

    # 3. URL extraction and analysis
    if urls_from_body is None:
        extracted_urls = extract_urls_from_text(body_text)
        html_urls = extract_urls_from_html(body_html)
        url_map: dict[str, dict[str, str]] = {}
        for u in extracted_urls:
            url_map[u] = {"url": u, "visible_text": ""}
        for u in html_urls:
            url_map[u["url"]] = u
        urls_for_analysis = list(url_map.values())
    else:
        urls_for_analysis = urls_from_body

    url_results = analyze_urls(
        urls_for_analysis,
        model_path=model_path,
        config=cfg,
        threshold=threshold,
    )

    # 4. Attachment analysis
    attachment_indicators = analyze_attachments(attachments)

    # 5. Risk scoring
    risk = calculate_risk_score(
        content_indicators,
        sender_indicators,
        url_results,
        attachment_indicators,
    )

    # 6. Safe actions
    safe_actions = _generate_safe_actions(risk["category"], url_results, attachment_indicators)

    # 7. Build result
    all_indicators = []
    all_indicators.extend(sender_indicators)
    all_indicators.extend(content_indicators)
    all_indicators.extend(attachment_indicators)
    for url_result in url_results:
        for ri in url_result.get("rule_indicators", []):
            all_indicators.append(
                {
                    "category": "url_rule",
                    "severity": "medium",
                    "indicator": ri,
                    "reason": f"URL: {url_result['url']}",
                    "url": url_result["url"],
                }
            )
        if url_result.get("ml_available") and url_result.get("ml_prediction") == "phishing":
            all_indicators.append(
                {
                    "category": "url_ml",
                    "severity": "high",
                    "indicator": "URL classified as phishing by ML model",
                    "reason": (
                        f"Probability: {url_result.get('ml_probability', 0):.2f}"
                    ),
                    "url": url_result["url"],
                }
            )

    return {
        "risk_score": risk["score"],
        "risk_category": risk["category"],
        "indicators": all_indicators,
        "urls": url_results,
        "attachments": attachments,
        "breakdown": risk["breakdown"],
        "total_indicators": risk["total_indicators"],
        "safe_actions": safe_actions,
    }


def _generate_safe_actions(
    category: str,
    url_results: list[dict[str, Any]],
    attachment_indicators: list[dict[str, Any]],
) -> list[str]:
    """Generate recommended safe actions based on analysis results."""
    actions = []

    if category == "high":
        actions.append("Do not click any links in this email")
        actions.append("Do not reply to this email")
        actions.append("Do not open or download any attachments")
        actions.append("Report this email to your IT security team")
        actions.append("Delete the email from your inbox")
    elif category == "medium":
        actions.append("Exercise caution with any links in this email")
        actions.append("Verify the sender through a separate channel")
        actions.append("Do not enter credentials on linked pages")
        actions.append("Consider reporting to your IT security team")
    else:
        actions.append("The email appears low-risk, but remain vigilant")
        actions.append("Verify unexpected requests through official channels")

    # URL-specific actions
    phishing_urls = [
        u for u in url_results
        if u.get("ml_available") and u.get("ml_prediction") == "phishing"
    ]
    if phishing_urls:
        actions.append(
            f"{len(phishing_urls)} link(s) classified as phishing - do not click"
        )

    # Attachment-specific actions
    risky_att = [
        i for i in attachment_indicators
        if i.get("category") == "risky_attachment"
    ]
    if risky_att:
        actions.append("Do not open any attachments - they may contain malware")

    # Deduplicate while preserving order
    seen = set()
    unique_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)

    return unique_actions


def escape_html(text: str) -> str:
    """Escape HTML for safe display."""
    return html_module.escape(text or "", quote=True)
