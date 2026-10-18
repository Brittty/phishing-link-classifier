# Phishing Link Classifier

An offline, modular phishing URL classifier built with Python and scikit-learn.
Extracts lexical and host-based features from raw URL strings, trains and compares
Logistic Regression and Random Forest models, and provides a CLI for prediction.

## Project Structure

```
phishing-link-classifier/
├── README.md
├── DATASET.md          # Dataset provenance: source, license, collection date, limitations
├── requirements.txt    # Runtime dependencies
├── requirements-dev.txt # Dev/test dependencies
├── setup.py            # Packaging
├── config.yaml         # Global config (paths, seeds, hyperparameters)
├── .gitignore          # Git ignore rules
├── data/
│   ├── raw/            # User-provided dataset downloads (git-ignored)
│   ├── processed/      # Featurized splits (git-ignored)
│   └── sample/         # Tracked small labeled sample for CI/smoke tests
├── models/             # Trained pipeline artifacts (git-ignored)
├── reports/            # Evaluation outputs, plots, metrics (git-ignored)
├── src/
│   ├── config.py       # Paths, constants, config loader
│   ├── dataset.py      # Loading, cleaning, leakage-safe splitting
│   ├── features.py     # Offline deterministic feature extractor
│   ├── model.py        # LR vs RF pipeline, training, save/load
│   ├── evaluate.py     # Metrics, validation-based thresholding, error analysis
│   ├── predict.py      # Inference helpers (single + batch)
│   ├── cli.py          # Command-line entry point
│   └── security.py     # Input validation, URL redaction, artifact trust
├── scripts/
│   └── train.py        # Thin orchestration wrapper
├── tests/              # Unit tests
└── notebooks/          # Optional EDA
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install runtime dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing)
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

## Dataset

See `DATASET.md` for provenance, license, and limitations.

A labeled CSV must have columns: `url` (string), `label` (int: 0=legitimate, 1=phishing).

A small tracked sample is provided at `data/sample/urls.csv` for smoke tests.

## Usage

### Train a model

```bash
python scripts/train.py --data data/raw/your_dataset.csv --out models/phishing_pipeline.joblib
```

### Predict a single URL

```bash
python -m src.cli predict --url "http://login-paypal.com.verify.xyz" --model models/phishing_pipeline.joblib
```

### Predict from a batch CSV file

```bash
python -m src.cli predict --input-file data/sample/urls.csv --model models/phishing_pipeline.joblib --output-file reports/predictions.csv --format csv
```

### Evaluate a model

```bash
python -m src.cli evaluate --data data/processed/test.csv --model models/phishing_pipeline.joblib
```

## Security

- Features are computed from the URL string only. **No URLs are visited, downloaded, or executed.**
- Query parameters are parsed for features but are not logged in plain text.
- Only load model artifacts from the local, trusted `models/` directory.

## Testing

```bash
pytest tests/
```

## Docker

Build the image:

```bash
docker build -t phishing-link-classifier .
```

Run a prediction:

```bash
docker run --rm phishing-link-classifier predict --url "https://example.com" --model models/phishing_pipeline.joblib --format json
```

Train inside the container (mount your data and models directories):

```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  phishing-link-classifier train --data data/raw/your_dataset.csv --out models/phishing_pipeline.joblib
```

## Continuous Integration

GitHub Actions runs the test suite on every push and pull request against Python 3.9, 3.10, 3.11, and 3.12.

## License

MIT