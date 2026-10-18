"""Unit tests for dataset loading and splitting."""

import os
import tempfile

import pandas as pd
import pytest

from src.dataset import load_dataset, train_val_test_split
from src.config import Config


def _write_csv(tmpdir, rows):
    path = os.path.join(tmpdir, "test.csv")
    df = pd.DataFrame(rows, columns=["url", "label"])
    df.to_csv(path, index=False)
    return path


def test_load_dataset_valid(tmp_path):
    path = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("http://phish.xyz", 1),
    ])
    df = load_dataset(path)
    assert len(df) == 2
    assert list(df.columns) == ["url", "label"]
    assert df["label"].dtype in (int,)


def test_load_dataset_missing_columns(tmp_path):
    path = os.path.join(tmp_path, "bad.csv")
    pd.DataFrame({"url": ["https://example.com"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_load_dataset_deduplication(tmp_path):
    path = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("https://example.com", 0),
        ("http://phish.xyz", 1),
    ])
    df = load_dataset(path)
    assert len(df) == 2  # duplicate removed


def test_load_dataset_conflicting_labels_removed(tmp_path):
    path = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("https://example.com", 1),  # conflict
        ("http://phish.xyz", 1),
    ])
    df = load_dataset(path)
    # Conflicting URL should be removed entirely
    assert len(df) == 1
    assert df.iloc[0]["url"] == "http://phish.xyz"


def test_train_val_test_split_sizes(tmp_path):
    config = Config.load()
    rows = [(f"https://site{i}.com/page{j}", i % 2) for i in range(20) for j in range(2)]
    path = _write_csv(tmp_path, rows)
    df = load_dataset(path)

    split = train_val_test_split(df, config)
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(df)

    # Proportions roughly match config
    train_frac = len(split.train) / len(df)
    assert abs(train_frac - config.split["train_size"]) < 0.15


def test_split_stratification_preserves_classes(tmp_path):
    config = Config.load()
    rows = [(f"https://site{i}.com", 0) for i in range(40)] + \
           [(f"http://phish{i}.xyz", 1) for i in range(10)]
    path = _write_csv(tmp_path, rows)
    df = load_dataset(path)

    split = train_val_test_split(df, config)
    # Each split should have both classes (given enough data)
    assert split.train["label"].nunique() == 2
    assert split.val["label"].nunique() == 2
    assert split.test["label"].nunique() == 2