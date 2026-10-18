"""Unit tests for CLI behavior using subprocess (argparse-based, not Click)."""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.cli import main


def _run_cli(args):
    """Run the CLI via subprocess to avoid Click CliRunner incompatibility."""
    cmd = [sys.executable, "-m", "src.cli"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    return result


def _write_csv(tmpdir, rows):
    path = os.path.join(tmpdir, "test.csv")
    pd.DataFrame(rows, columns=["url", "label"]).to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_model(tmp_path):
    """Train a tiny model for CLI tests."""
    from src.model import build_pipeline, train_model, save_pipeline
    from src.config import Config

    config = Config.load()
    X = ["https://example.com"] * 8 + ["http://evil.xyz"] * 8
    y = [0] * 8 + [1] * 8
    result = train_model(X, y, model_type="logistic_regression", config=config)

    model_path = os.path.join(tmp_path, "model.joblib")
    save_pipeline(result.pipeline, model_path)
    return model_path


def test_cli_predict_single_url(tiny_model):
    result = _run_cli([
        "predict", "--url", "https://example.com",
        "--model", tiny_model,
        "--format", "json",
    ])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "prediction" in data
    assert "probability" in data


def test_cli_predict_invalid_url(tiny_model):
    result = _run_cli([
        "predict", "--url", "",
        "--model", tiny_model,
    ])
    assert result.returncode != 0


def test_cli_predict_missing_model(tmp_path):
    result = _run_cli([
        "predict", "--url", "https://example.com",
        "--model", os.path.join(tmp_path, "nonexistent.joblib"),
    ])
    assert result.returncode != 0


def test_cli_predict_batch_csv(tiny_model, tmp_path):
    input_csv = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("http://evil.xyz", 1),
    ])
    output_csv = os.path.join(tmp_path, "out.csv")

    result = _run_cli([
        "predict", "--input-file", input_csv,
        "--model", tiny_model,
        "--output-file", output_csv,
    ])
    assert result.returncode == 0

    df = pd.read_csv(output_csv)
    assert "url" in df.columns
    assert "prediction" in df.columns
    assert "probability" in df.columns


def test_cli_train(tmp_path):
    data_csv = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("http://evil.xyz", 1),
        ("https://google.com", 0),
        ("http://phish.bid", 1),
        ("https://wikipedia.org", 0),
        ("http://scam.xyz", 1),
    ])
    model_out = os.path.join(tmp_path, "trained.joblib")

    result = _run_cli([
        "train", "--data", data_csv, "--out", model_out,
        "--model-type", "logistic_regression",
    ])
    assert result.returncode == 0
    assert os.path.exists(model_out)


def test_cli_compare(tmp_path):
    data_csv = _write_csv(tmp_path, [
        ("https://example.com", 0),
        ("http://evil.xyz", 1),
        ("https://google.com", 0),
        ("http://phish.bid", 1),
        ("https://wikipedia.org", 0),
        ("http://scam.xyz", 1),
    ])
    model_out = os.path.join(tmp_path, "trained.joblib")

    result = _run_cli([
        "train", "--data", data_csv, "--out", model_out,
        "--compare",
    ])
    assert result.returncode == 0