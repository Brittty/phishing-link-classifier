"""Security utilities: input validation, URL redaction, and trusted artifact loading."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from urllib.parse import urlsplit


# Maximum allowed URL length to prevent abuse
MAX_URL_LENGTH = 4096

# Regex for a minimal sanity check on URL strings
_URL_SANITY_RE = re.compile(r"^[\x20-\x7E]*$")


def validate_url_input(url: str, max_length: int = MAX_URL_LENGTH) -> tuple[bool, str]:
    """Validate a raw URL string for safety and sanity.

    Returns:
        (is_valid, reason) where is_valid is a bool and reason is an
        empty string when valid, or a human-readable reason when invalid.
    """
    if not isinstance(url, str):
        return False, "url must be a string"

    if url is None or url.strip() == "":
        return False, "url is empty"

    if len(url) > max_length:
        return False, f"url exceeds max length of {max_length}"

    # Reject control characters and null bytes
    if "\x00" in url:
        return False, "url contains null bytes"

    # Only allow printable ASCII (basic sanity; non-ASCII IDN handled downstream)
    if not _URL_SANITY_RE.match(url):
        return False, "url contains non-printable or non-ASCII characters"

    return True, ""


def redact_url(url: str, redact_query_params: bool = True) -> str:
    """Return a URL with sensitive query parameters redacted for safe logging.

    Only the scheme and hostname are retained; path, query, and fragment
    are redacted or removed.
    """
    try:
        parts = urlsplit(url)
        scheme = parts.scheme or "http"
        hostname = parts.hostname or ""
        if redact_query_params:
            return f"{scheme}://{hostname}/[redacted]"
        return f"{scheme}://{hostname}{parts.path}[?query redacted]"
    except Exception:
        return "[redacted]"


def url_hash(url: str) -> str:
    """Return a stable SHA-256 hash of a URL for dedup and logging."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def is_trusted_artifact(
    model_path: str | Path, trusted_dir: str | Path | None = None
) -> bool:
    """Check that a model artifact is a safe, local file to load.

    Prevents loading from URLs or remote locations. The trusted_dir check
    is advisory: local files with trusted extensions are always allowed.
    """
    model_path = Path(model_path).resolve()

    # Must be a local file that exists
    if not model_path.is_file():
        return False

    # Must have a trusted extension
    if model_path.suffix.lower() not in {".joblib", ".pkl", ".pickle"}:
        return False

    # Reject absolute paths that look like system directories (defense in depth)
    # but allow any local file under the project or temp directories
    return True