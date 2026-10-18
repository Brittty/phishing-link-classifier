"""Dataset loading, cleaning, and leakage-safe splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_dataset(csv_path: str, url_column: str = "url", label_column: str = "label") -> pd.DataFrame:
    """Load a labeled URL CSV and validate required columns.

    Args:
        csv_path: Path to the CSV file.
        url_column: Name of the URL column.
        label_column: Name of the label column.

    Returns:
        DataFrame with columns [url_column, label_column].

    Raises:
        ValueError: If required columns are missing or the file cannot be read.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at {csv_path}: {exc}") from exc

    missing = [c for c in (url_column, label_column) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only required columns
    df = df[[url_column, label_column]].copy()

    # Drop rows with missing url or label
    df = df.dropna(subset=[url_column, label_column])

    # Ensure labels are integers
    df[label_column] = df[label_column].astype(int)

    # Ensure URLs are strings
    df[url_column] = df[url_column].astype(str)

    # Drop empty URLs
    df = df[df[url_column].str.strip() != ""]

    # Resolve conflicting labels: same URL with both 0 and 1 -> drop entirely
    # Check conflicts BEFORE dedup so conflicting URLs are fully removed
    conflicts = (
        df.groupby(url_column)[label_column]
        .nunique()
        .rename("n_labels")
        .reset_index()
    )
    conflicting_urls = set(conflicts[conflicts["n_labels"] > 1][url_column])
    before = len(df)
    df = df[~df[url_column].isin(conflicting_urls)]
    dropped_conflicts = before - len(df)

    # Drop duplicate URLs (exact same url + label)
    before = len(df)
    df = df.drop_duplicates(subset=[url_column, label_column])
    dropped_dups = before - len(df)

    df = df.reset_index(drop=True)

    return df


def _extract_registered_domain(url: str) -> str:
    """Extract the registered domain from a URL for grouped splitting."""
    import tldextract

    ext = tldextract.extract(url)
    registered = (
        getattr(ext, "top_domain_under_public_suffix", None)
        or getattr(ext, "registered_domain", "")
        or ext.domain
        or ""
    )
    return registered


def train_val_test_split(
    df: pd.DataFrame,
    config: Config,
    url_column: str | None = None,
    label_column: str | None = None,
) -> SplitResult:
    """Stratified, optional grouped-by-domain train/validation/test split.

    The split respects class proportions and (optionally) keeps all URLs from
    the same registered domain in a single split to prevent leakage.

    Test data is isolated and never used for threshold selection or tuning.
    """
    url_column = url_column or config.url_column
    label_column = label_column or config.label_column

    train_size = config.split["train_size"]
    val_size = config.split["val_size"]
    test_size = config.split["test_size"]
    seed = config.seed
    stratify = config.split.get("stratify", True)
    group_by_domain = config.split.get("group_by_domain", True)

    # Validate split proportions sum to 1
    total = train_size + val_size + test_size
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    y = df[label_column].values if stratify else None

    # First split: train vs (val + test)
    train_val_size = train_size + val_size
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=y if stratify else None,
    )

    # Second split: val vs test from the temp portion
    # Adjust val fraction within the temp set
    val_fraction_within_temp = val_size / (val_size + test_size)
    y_temp = temp_df[label_column].values if stratify else None

    if group_by_domain:
        # Group-based split: ensure domains don't leak across val/test
        temp_df = temp_df.copy()
        temp_df["_domain"] = temp_df[url_column].apply(_extract_registered_domain)

        # Get unique domains with their labels (majority label per domain)
        domain_labels = (
            temp_df.groupby("_domain")[label_column]
            .agg(lambda x: x.mode()[0])
            .reset_index()
        )

        # Fall back to non-stratified if too few domains per class
        use_stratify = stratify
        try:
            domain_train, domain_test = train_test_split(
                domain_labels,
                train_size=val_fraction_within_temp,
                random_state=seed,
                stratify=domain_labels[label_column] if use_stratify else None,
            )
        except ValueError:
            # Not enough members per class; retry without stratification
            domain_train, domain_test = train_test_split(
                domain_labels,
                train_size=val_fraction_within_temp,
                random_state=seed,
                stratify=None,
            )

        val_domains = set(domain_train["_domain"])
        test_domains = set(domain_test["_domain"])

        val_df = temp_df[temp_df["_domain"].isin(val_domains)].drop(columns=["_domain"])
        test_df = temp_df[temp_df["_domain"].isin(test_domains)].drop(columns=["_domain"])
    else:
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_fraction_within_temp,
            random_state=seed,
            stratify=y_temp if stratify else None,
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return SplitResult(train=train_df, val=val_df, test=test_df)