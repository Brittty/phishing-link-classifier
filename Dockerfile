FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml ./
COPY setup.py ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for runtime data
RUN mkdir -p data/raw data/processed data/sample models reports/figures

# Default command: show help
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]