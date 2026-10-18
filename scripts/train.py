"""Thin orchestration wrapper for end-to-end training.

Usage:
    python scripts/train.py --data data/raw/dataset.csv --out models/phishing_pipeline.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import main  # noqa: E402


if __name__ == "__main__":
    # Prepend the 'train' subcommand and pass remaining args
    sys.exit(main(["train"] + sys.argv[1:]))