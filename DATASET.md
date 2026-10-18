# Dataset Provenance

## Required Format

The classifier expects a labeled CSV file with raw URL strings and binary labels:

| Column   | Type   | Description                      |
|----------|--------|----------------------------------|
| `url`    | string | The full URL to classify         |
| `label`  | int    | 0 = legitimate, 1 = phishing     |

The loader **requires** raw URL strings. Precomputed-feature datasets without a `url` column will be rejected.

## Suggested Public Sources

### Phishing URLs (label = 1)

| Source | URL | License | Notes |
|--------|-----|---------|-------|
| PhishTank | https://www.phishtank.com | Creative Commons Attribution 3.0 | Raw phishing URLs; free API key or CSV download required |
| OpenPhish | https://openphish.com | Free for research; commercial restrictions possible | Alternative or supplemental phishing feed |

### Legitimate URLs (label = 0)

| Source | URL | License | Notes |
|--------|-----|---------|-------|
| Tranco Top-1M | https://tranco-list.org | Open | Raw domain names; convert to `https://<domain>/` |
| Majestic Million | https://majestic.com | Open | Raw domain names |

## Documentation Requirements

When using a dataset, record the following in this file:

- **Source URL(s)**: Where each dataset was downloaded
- **Collection date**: When the data was collected (e.g., "2026-01-01 to 2026-09-30")
- **License**: The license for each source
- **Limitations**: Known issues (e.g., phishing URLs may be taken down quickly, legitimate lists may contain parked domains, class imbalance is expected)

## Example Entry

```
Source: PhishTank (phishing) + Tranco Top-1M (legitimate)
Collection date: 2026-09-30
License: PhishTank CC BY 3.0; Tranco Open
Limitations: PhishTank URLs may be inactive by the time of training; class imbalance expected (~1:4 phishing:legitimate)
```

## Sample Dataset

A small tracked sample is provided at `data/sample/urls.csv` for CI and smoke testing only. It is not sufficient for training a production model.