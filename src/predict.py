"""Prediction helpers: single URL and batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .model import load_pipeline
from .security import validate_url_input, redact_url, url_hash, is_trusted_artifact


def predict_url(
    pipeline: Any,
    url: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict on a single URL string.

    Returns a dict with the URL (redacted), hash, label, probability, and
    whether the input was valid.
    """
    is_valid, reason = validate_url_input(url)
    if not is_valid:
        return {
            "url": str(url) if url is not None else "",
            "url_hash": "",
            "error": reason,
            "is_valid": False,
        }

    probs = pipeline.predict_proba([url])[0]
    phishing_prob = float(probs[1])
    label = 1 if phishing_prob >= threshold else 0

    return {
        "url": redact_url(url),
        "url_hash": url_hash(url),
        "is_valid": True,
        "probability": round(phishing_prob, 4),
        "label": int(label),
        "prediction": "phishing" if label == 1 else "legitimate",
        "threshold": threshold,
    }


def predict_batch(
    pipeline: Any,
    urls: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Predict on a batch of URL strings."""
    return [predict_url(pipeline, u, threshold=threshold) for u in urls]


def predict_csv(
    pipeline: Any,
    input_csv: str | Path,
    url_column: str = "url",
    threshold: float = 0.5,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Predict on URLs in a CSV file and optionally write a report.

    The output CSV contains: url (redacted), url_hash, label, probability,
    prediction, threshold.
    """
    df = pd.read_csv(input_csv)

    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in {input_csv}")

    results = predict_batch(pipeline, df[url_column].tolist(), threshold=threshold)

    out_df = pd.DataFrame(results)
    # Reorder columns for readability
    col_order = [
        "url", "url_hash", "is_valid", "probability",
        "label", "prediction", "threshold", "error",
    ]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    return out_df


def load_model_for_prediction(
    model_path: str | Path,
    trusted_dir: str | Path | None = None,
) -> Any:
    """Load a model artifact after verifying it is a trusted local file."""
    if not is_trusted_artifact(model_path, trusted_dir=trusted_dir):
        raise ValueError(
            f"Model artifact is not trusted or does not exist: {model_path}. "
            f"Only load .joblib/.pkl files from the trusted models directory."
        )
    return load_pipeline(model_path)