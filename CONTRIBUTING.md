# Contributing

## Development Setup

1. Use Python 3.10 or newer.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Place a locally downloaded `creditcard.csv` beside `fraud_detection.py` when running the full workflow.

## Before Opening A Pull Request

Run the following check:

```powershell
python -m py_compile fraud_detection.py
```

Do not commit `creditcard.csv`, generated `artifacts/`, notebook checkpoints, or credentials.

## Modeling Rules

- Split data before fitting transformations or sampling.
- Never use test data to select features, hyperparameters, or decision thresholds.
- Report PR-AUC and the fraud-class precision/recall, not accuracy alone.
- Keep random seeds explicit so experiments can be reproduced.
