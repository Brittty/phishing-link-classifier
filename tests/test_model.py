"""Unit tests for model training, save/load, and prediction."""

import os
import tempfile

import joblib
import numpy as np
import pytest

from src.config import Config
from src.model import build_pipeline, train_model, save_pipeline, load_pipeline, compare_models


def test_build_pipeline_logistic_regression():
    pipe = build_pipeline(model_type="logistic_regression")
    assert pipe is not None

    X = ["https://example.com", "http://evil.xyz"]
    y = [0, 1]
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (2, 2)


def test_build_pipeline_random_forest():
    pipe = build_pipeline(model_type="random_forest")
    assert pipe is not None

    X = ["https://example.com", "http://evil.xyz"]
    y = [0, 1]
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (2, 2)


def test_train_model_output_probs_in_range():
    config = Config.load()
    X = ["https://example.com"] * 10 + ["http://evil.xyz"] * 10
    y = [0] * 10 + [1] * 10

    result = train_model(X, y, model_type="logistic_regression", config=config)
    probs = result.pipeline.predict_proba(X)[:, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_save_and_load_pipeline(tmp_path):
    config = Config.load()
    X = ["https://example.com", "http://evil.xyz"]
    y = [0, 1]

    result = train_model(X, y, model_type="logistic_regression", config=config)

    model_path = os.path.join(tmp_path, "test_pipeline.joblib")
    save_pipeline(result.pipeline, model_path)

    loaded = load_pipeline(model_path)
    probs_original = result.pipeline.predict_proba(X)[:, 1]
    probs_loaded = loaded.predict_proba(X)[:, 1]
    np.testing.assert_array_almost_equal(probs_original, probs_loaded)


def test_compare_models(tmp_path):
    config = Config.load()
    X = ["https://example.com"] * 20 + ["http://evil.xyz"] * 20
    y = [0] * 20 + [1] * 20

    results = compare_models(X, y, X, y, config=config)
    assert "logistic_regression" in results
    assert "random_forest" in results

    for model_type, res in results.items():
        assert res.pipeline is not None
        probs = res.pipeline.predict_proba(X)[:, 1]
        assert np.all((probs >= 0.0) & (probs <= 1.0))