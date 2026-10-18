"""Configuration loading and path constants for the phishing link classifier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    """Project path constants resolved relative to the project root."""

    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    sample_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any], root: Path) -> "Paths":
        p = config["paths"]
        return cls(
            root=root,
            data_dir=root / p["data_dir"],
            raw_dir=root / p["raw_dir"],
            processed_dir=root / p["processed_dir"],
            sample_dir=root / p["sample_dir"],
            models_dir=root / p["models_dir"],
            reports_dir=root / p["reports_dir"],
            figures_dir=root / p["figures_dir"],
        )


@dataclass
class Config:
    """Full application configuration loaded from config.yaml."""

    paths: Paths
    seed: int
    feature_schema: str
    dataset: dict[str, Any]
    split: dict[str, Any]
    models: dict[str, Any]
    search: dict[str, Any]
    threshold: dict[str, Any]
    misleading_brands: list[str]
    suspicious_tlds: list[str]
    common_tlds: list[str]
    security: dict[str, Any]

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        """Load configuration from a YAML file."""
        if config_path is None:
            # Default: config.yaml next to the project root
            config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        root = config_path.parent
        paths = Paths.from_config(raw, root)

        return cls(
            paths=paths,
            seed=raw["seed"],
            feature_schema=raw["feature_schema"],
            dataset=raw["dataset"],
            split=raw["split"],
            models=raw["models"],
            search=raw["search"],
            threshold=raw["threshold"],
            misleading_brands=raw["misleading_brands"],
            suspicious_tlds=raw["suspicious_tlds"],
            common_tlds=raw["common_tlds"],
            security=raw["security"],
        )

    @property
    def url_column(self) -> str:
        return self.dataset["url_column"]

    @property
    def label_column(self) -> str:
        return self.dataset["label_column"]

    @property
    def legitimate_label(self) -> int:
        return self.dataset["legitimate_label"]

    @property
    def phishing_label(self) -> int:
        return self.dataset["phishing_label"]

    @property
    def max_url_length(self) -> int:
        return self.security["max_url_length"]

    @property
    def redact_query_params(self) -> bool:
        return self.security["redact_query_params"]

    @property
    def trusted_model_dir(self) -> Path:
        return self.paths.models_dir


def load_config(config_path: str | Path | None = None) -> Config:
    """Convenience wrapper to load Config from a YAML path."""
    return Config.load(config_path)