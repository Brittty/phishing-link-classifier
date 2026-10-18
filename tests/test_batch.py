"""Unit tests for batch CSV prediction."""

import os
import tempfile

import pandas as pd
import pytest

from src.config import Config
from src.model import build_pipeline, train_model, save_pipeline
from src.predict import predict_csv


def test_predict_csv_writes_report(tmp_path):
    config = Config.load()
    X = ["https://example.com"] * 6 + ["http://evil.xyz"] * 6
    y = [0] * 6 + [1] * 6

    result = train_model(X, y, model_type="logistic_regression", config=config)

    model_path = os.path.join(tmp_path, "model.joblib")
    save_pipeline(result.pipeline, model_path)

    input_csv = os.path.join(tmp_path, "input.csv")
    pd.DataFrame({"url": ["https://example.com", "http://evil.xyz", ""]}).to_csv(input_csv, index=False)

    output_csv = os.path.join(tmp_path, "output.csv")
    df = predict_csv(result.pipeline, input_csv, url_column="url", output_csv=output_csv)

    assert len(df) == 3
    assert "url" in df.columns
    assert "prediction" in df.columns
    assert "probability" in df.columns

    # Invalid URL (empty) should have is_valid=False
    empty_rows = df[df["url"].fillna("") == ""]
    if len(empty_rows) > 0:
        assert empty_rows.iloc[0]["is_valid"] is False


def test_predict_csv_redacts_query_params(tmp_path):
    config = Config.load()
    X = ["https://example.com"] * 4 + ["http://evil.xyz"] * 4
    y = [0] * 4 + [1] * 4

    result = train_model(X, y, model_type="logistic_regression", config=config)

    input_csv = os.path.join(tmp_path, "input.csv")
    pd.DataFrame({"url": ["https://example.com/page?token=secret123"]}).to_csv(input_csv, index=False)

    output_csv = os.path.join(tmp_path, "output.csv")
    df = predict_csv(result.pipeline, input_csv, url_column="url", output_csv=output_csv)

    url_value = df.iloc[0]["url"]
    assert "secret123" not in str(url_value)
    assert "[redacted]" in str(url_value)