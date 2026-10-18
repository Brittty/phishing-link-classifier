"""Command-line interface for the phishing link classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config, Config
from .dataset import load_dataset, train_val_test_split
from .evaluate import (
    evaluate_model,
    select_threshold_on_validation,
    error_analysis,
    cross_validate_model,
)
from .features import FEATURE_NAMES
from .model import build_pipeline, train_model, compare_models, save_pipeline, load_pipeline
from .predict import (
    predict_url,
    predict_csv,
    load_model_for_prediction,
)
from .security import validate_url_input, is_trusted_artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline phishing URL classifier (scikit-learn)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- predict ----
    pred = subparsers.add_parser("predict", help="Predict phishing probability for URLs")
    pred.add_argument("--url", type=str, help="Single URL to predict")
    pred.add_argument(
        "--input-file", type=str, help="CSV file with a 'url' column for batch prediction"
    )
    pred.add_argument(
        "--output-file", type=str, help="Output CSV path for batch predictions"
    )
    pred.add_argument(
        "--format", type=str, choices=["csv", "json"], default="csv",
        help="Output format for single prediction (csv/json)"
    )
    pred.add_argument(
        "--model", type=str, default="models/phishing_pipeline.joblib",
        help="Path to trained pipeline artifact"
    )
    pred.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default: from config or 0.5)"
    )

    # ---- train ----
    train = subparsers.add_parser("train", help="Train a model pipeline")
    train.add_argument("--data", type=str, required=True, help="Path to labeled CSV")
    train.add_argument(
        "--out", type=str, default="models/phishing_pipeline.joblib",
        help="Output path for the trained pipeline"
    )
    train.add_argument(
        "--model-type", type=str, choices=["logistic_regression", "random_forest"],
        default="logistic_regression", help="Model type to train"
    )
    train.add_argument(
        "--tune", action="store_true", help="Enable hyperparameter tuning"
    )
    train.add_argument(
        "--compare", action="store_true",
        help="Train and compare both model types"
    )

    # ---- evaluate ----
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_p.add_argument(
        "--data", type=str, required=True, help="Path to test CSV"
    )
    eval_p.add_argument(
        "--model", type=str, required=True, help="Path to trained pipeline artifact"
    )
    eval_p.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory for metrics and plots"
    )
    eval_p.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default: use validation-selected threshold)"
    )

    return parser


def cmd_predict(args: argparse.Namespace, config: Config) -> int:
    model_path = args.model
    if not is_trusted_artifact(model_path, trusted_dir=config.trusted_model_dir):
        print(f"ERROR: Untrusted or missing model artifact: {model_path}", file=sys.stderr)
        return 1

    pipeline = load_pipeline(model_path)

    threshold = args.threshold
    if threshold is None:
        threshold = config.threshold.get("default", 0.5)

    if args.url:
        is_valid, reason = validate_url_input(args.url)
        if not is_valid:
            print(f"ERROR: Invalid URL: {reason}", file=sys.stderr)
            return 1

        result = predict_url(pipeline, args.url, threshold=threshold)

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"URL: {result['url']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability (phishing): {result['probability']}")
            print(f"Threshold: {result['threshold']}")

    elif args.input_file:
        df = predict_csv(
            pipeline,
            args.input_file,
            url_column=config.url_column,
            threshold=threshold,
            output_csv=args.output_file,
        )
        if args.output_file:
            print(f"Batch predictions written to {args.output_file}")
        else:
            print(df.to_json(orient="records", indent=2))

    else:
        print("ERROR: Either --url or --input-file is required for predict", file=sys.stderr)
        return 1

    return 0


def cmd_train(args: argparse.Namespace, config: Config) -> int:
    df = load_dataset(
        args.data,
        url_column=config.url_column,
        label_column=config.label_column,
    )
    print(f"Loaded {len(df)} labeled URLs after cleaning")

    split = train_val_test_split(df, config)
    print(
        f"Split: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}"
    )

    X_train = split.train[config.url_column].tolist()
    y_train = split.train[config.label_column].tolist()
    X_val = split.val[config.url_column].tolist()
    y_val = split.val[config.label_column].tolist()
    X_test = split.test[config.url_column].tolist()
    y_test = split.test[config.label_column].tolist()

    if args.compare:
        results = compare_models(X_train, y_train, X_val, y_val, config=config)
        for model_type, result in results.items():
            print(
                f"{model_type}: best_cv_score={result.best_cv_score:.4f}, "
                f"best_params={result.best_params}"
            )
        best_type = max(results, key=lambda k: results[k].best_cv_score)
        pipeline = results[best_type].pipeline
        print(f"Selected best model: {best_type}")
    else:
        result = train_model(
            X_train, y_train,
            model_type=args.model_type,
            config=config,
            tune_hyperparameters=args.tune,
        )
        pipeline = result.pipeline
        print(
            f"Trained {args.model_type}: "
            f"best_cv_score={result.best_cv_score:.4f}, "
            f"params={result.best_params}"
        )

    best_threshold, threshold_results = select_threshold_on_validation(
        pipeline, X_val, y_val,
        metric=config.threshold.get("metric", "f1"),
    )
    print(f"Selected threshold on validation: {best_threshold} "
          f"(metric={config.threshold.get('metric', 'f1')})")

    metrics = evaluate_model(
        pipeline, X_test, y_test,
        threshold=best_threshold,
        output_dir=Path(config.paths.reports_dir),
        config=config,
    )
    print(
        f"Test metrics: accuracy={metrics.accuracy}, precision={metrics.precision}, "
        f"recall={metrics.recall}, f1={metrics.f1}, pr_auc={metrics.pr_auc}, "
        f"roc_auc={metrics.roc_auc}"
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pipeline(pipeline, output_path)
    print(f"Model saved to {output_path}")

    return 0


def cmd_evaluate(args: argparse.Namespace, config: Config) -> int:
    if not is_trusted_artifact(args.model, trusted_dir=config.trusted_model_dir):
        print(f"ERROR: Untrusted or missing model artifact: {args.model}", file=sys.stderr)
        return 1

    pipeline = load_pipeline(args.model)

    df = load_dataset(
        args.data,
        url_column=config.url_column,
        label_column=config.label_column,
    )
    X_test = df[config.url_column].tolist()
    y_test = df[config.label_column].tolist()

    threshold = args.threshold
    if threshold is None:
        threshold = config.threshold.get("default", 0.5)
        print(f"Using default threshold: {threshold}")

    metrics = evaluate_model(
        pipeline, X_test, y_test,
        threshold=threshold,
        output_dir=Path(args.output_dir),
        config=config,
    )
    print(
        f"Test metrics: accuracy={metrics.accuracy}, precision={metrics.precision}, "
        f"recall={metrics.recall}, f1={metrics.f1}, pr_auc={metrics.pr_auc}, "
        f"roc_auc={metrics.roc_auc}"
    )
    print(f"Confusion matrix: {metrics.confusion_matrix}")
    print(f"Metrics written to {args.output_dir}/metrics.json")

    errors = error_analysis(
        pipeline, X_test, y_test, threshold=threshold, config=config
    )
    if len(errors) > 0:
        errors_path = Path(args.output_dir) / "error_analysis.csv"
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(errors_path, index=False)
        print(f"Error analysis written to {errors_path} ({len(errors)} errors)")
    else:
        print("No misclassifications on test set.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config()

    if args.command == "predict":
        return cmd_predict(args, config)
    elif args.command == "train":
        return cmd_train(args, config)
    elif args.command == "evaluate":
        return cmd_evaluate(args, config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())