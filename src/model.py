"""Model pipeline construction, training, and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import FeatureExtractor, FEATURE_NAMES


@dataclass
class TrainResult:
    pipeline: Pipeline
    best_params: dict[str, Any]
    best_cv_score: float
    model_type: str


def build_pipeline(
    model_type: str = "logistic_regression",
    config: Config | None = None,
) -> Pipeline:
    """Build a scikit-learn Pipeline: FeatureExtractor -> scaler -> classifier."""
    cfg = config or Config.load()
    model_cfg = cfg.models.get(model_type, {})

    extractor = FeatureExtractor(
        feature_names=FEATURE_NAMES,
        config={
            "misleading_brands": cfg.misleading_brands,
            "suspicious_tlds": cfg.suspicious_tlds,
            "common_tlds": cfg.common_tlds,
            "max_url_length": cfg.max_url_length,
        },
    )

    scaler = StandardScaler(with_mean=False)

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            solver=model_cfg.get("solver", "liblinear"),
            class_weight=model_cfg.get("class_weight", "balanced"),
            max_iter=model_cfg.get("max_iter", 1000),
            random_state=model_cfg.get("random_seed", cfg.seed),
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 300),
            class_weight=model_cfg.get("class_weight", "balanced_subsample"),
            max_depth=model_cfg.get("max_depth"),
            min_samples_split=model_cfg.get("min_samples_split", 2),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 1),
            random_state=model_cfg.get("random_seed", cfg.seed),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("features", extractor),
            ("scaler", scaler),
            ("clf", clf),
        ]
    )
    return pipeline


def train_model(
    X_train: Any,
    y_train: Any,
    model_type: str = "logistic_regression",
    config: Config | None = None,
    tune_hyperparameters: bool = False,
) -> TrainResult:
    """Train a model pipeline on the training data."""
    cfg = config or Config.load()

    if tune_hyperparameters:
        pipeline = build_pipeline(model_type=model_type, config=cfg)

        if model_type == "logistic_regression":
            param_grid = {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": ["balanced", None],
            }
        else:
            param_grid = {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            }

        n_samples = len(y_train)
        cv_folds = cfg.search.get("cv_folds", 5)
        # Reduce CV folds for small datasets
        cv_folds = min(cv_folds, max(2, n_samples // 2))

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=cfg.seed,
        )

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=cfg.search.get("scoring", "f1"),
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        return TrainResult(
            pipeline=search.best_estimator_,
            best_params=search.best_params_,
            best_cv_score=search.best_score_,
            model_type=model_type,
        )
    else:
        pipeline = build_pipeline(model_type=model_type, config=cfg)
        pipeline.fit(X_train, y_train)
        return TrainResult(
            pipeline=pipeline,
            best_params={},
            best_cv_score=float("nan"),
            model_type=model_type,
        )


def save_pipeline(pipeline: Pipeline, model_path: str | Path) -> None:
    """Save a fitted pipeline to disk using joblib."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


def load_pipeline(model_path: str | Path) -> Pipeline:
    """Load a fitted pipeline from disk using joblib."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def compare_models(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    config: Config | None = None,
) -> dict[str, TrainResult]:
    """Train Logistic Regression and Random Forest and return both results."""
    cfg = config or Config.load()
    results: dict[str, TrainResult] = {}

    for model_type in ("logistic_regression", "random_forest"):
        result = train_model(
            X_train,
            y_train,
            model_type=model_type,
            config=cfg,
            tune_hyperparameters=True,
        )
        results[model_type] = result

    return results