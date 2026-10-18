"""Evaluation: metrics, validation-based thresholding, error analysis, plots."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import Config
from .features import FEATURE_NAMES
from .model import build_pipeline, load_pipeline
from .security import redact_url, url_hash


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    threshold: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _precision(y_true, y_pred, pos_label=1, zero_division=0):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)


def _recall(y_true, y_pred, pos_label=1, zero_division=0):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)


def _f1(y_true, y_pred, pos_label=1, zero_division=0):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)


def select_threshold_on_validation(
    pipeline: Any,
    X_val: Any,
    y_val: Any,
    metric: str = "f1",
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, dict[float, dict[str, float]]]:
    """Choose a decision threshold on validation data only.

    The test set is never used here.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [round(t / 20, 2) for t in range(1, 20)]

    val_probs = pipeline.predict_proba(X_val)[:, 1]

    results: dict[float, dict[str, float]] = {}
    best_threshold = 0.5
    best_score = -1.0

    for th in candidate_thresholds:
        y_pred = (val_probs >= th).astype(int)
        precision = _precision(y_val, y_pred, pos_label=1, zero_division=0)
        recall = _recall(y_val, y_pred, pos_label=1, zero_division=0)
        f1 = _f1(y_val, y_pred, pos_label=1, zero_division=0)

        results[th] = {"precision": precision, "recall": recall, "f1": f1}

        if metric == "f1":
            score = f1
        elif metric == "precision":
            score = precision
        elif metric == "recall":
            score = recall
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = th

    return best_threshold, results


def evaluate_model(
    pipeline: Any,
    X_test: Any,
    y_test: Any,
    threshold: float = 0.5,
    output_dir: str | Path | None = None,
    config: Config | None = None,
) -> Metrics:
    """Evaluate a fitted pipeline on held-out test data."""
    cfg = config or Config.load()

    probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = _precision(y_test, y_pred, pos_label=1, zero_division=0)
    recall = _recall(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = _f1(y_test, y_pred, pos_label=1, zero_division=0)

    pr_auc = average_precision_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_list = cm.tolist()

    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["legitimate", "phishing"],
        output_dict=True, zero_division=0,
    )

    metrics = Metrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        threshold=threshold,
        confusion_matrix=cm_list,
        classification_report=report,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        _save_plots(probs, y_test, y_pred, output_dir)

    return metrics


def _save_plots(probs: np.ndarray, y_test: Any, y_pred: np.ndarray, output_dir: Path) -> None:
    """Save PR curve, ROC curve, and confusion matrix heatmap."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall_curve, precision_curve, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pr_curve.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Legitimate", "Phishing"])
    plt.yticks([0, 1], ["Legitimate", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png")
    plt.close()


def cross_validate_model(
    pipeline: Any,
    X: Any,
    y: Any,
    config: Config | None = None,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Run stratified cross-validation and return mean scores."""
    cfg = config or Config.load()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.seed)

    scores = {}
    for scoring in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        cv_scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        scores[f"cv_mean_{scoring}"] = round(float(cv_scores.mean()), 4)
        scores[f"cv_std_{scoring}"] = round(float(cv_scores.std()), 4)

    return scores


def error_analysis(
    pipeline: Any,
    X_test: Any,
    y_test: Any,
    threshold: float = 0.5,
    config: Config | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of misclassified URLs with probabilities."""
    cfg = config or Config.load()

    probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    urls = list(X_test)
    y_test_list = list(y_test)

    rows = []
    for url, true_label, pred_label, prob in zip(urls, y_test_list, y_pred, probs):
        if true_label != pred_label:
            error_type = (
                "false_positive"
                if true_label == 0 and pred_label == 1
                else "false_negative"
            )
            rows.append(
                {
                    "url": redact_url(url, redact_query_params=cfg.redact_query_params),
                    "url_hash": url_hash(url),
                    "true_label": int(true_label),
                    "true_class": "legitimate" if true_label == 0 else "phishing",
                    "predicted_label": int(pred_label),
                    "predicted_class": "legitimate" if pred_label == 0 else "phishing",
                    "probability": round(float(prob), 4),
                    "error_type": error_type,
                }
            )

    return pd.DataFrame(rows)