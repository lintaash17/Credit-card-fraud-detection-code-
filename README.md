# Credit Card Fraud Detection

Leakage-free fraud-model training and evaluation for the Kaggle credit-card transaction dataset. The project is built around the fact that fraud is a rare event: it preserves the real class distribution in validation and test data, evaluates with precision-recall metrics, and selects thresholds without looking at the test set.

## Project Structure

| Path | Purpose |
| --- | --- |
| `fraud_detection.py` | Reproducible training, model comparison, and evaluation entry point. |
| `requirements.txt` | Python dependencies. |
| `artifacts/` | Generated evaluation charts and model comparison CSV; created at runtime and not committed. |
| `4th semester project data science .ipynb` | Original project notebook, retained for reference only. |

## Setup

1. Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Put `creditcard.csv` in this project directory. The dataset is intentionally not committed because of its size.
3. Install dependencies and run the script:

```powershell
python -m pip install -r requirements.txt
python fraud_detection.py
```

## What The Script Does

- Validates the expected dataset shape and removes exact duplicate records.
- Makes stratified train, validation, and test partitions before any fitting.
- Fits scaling only inside a scikit-learn pipeline, using training data only.
- Keeps validation and test data at their original fraud rate. It does not duplicate fraud records into them.
- Compares class-weighted Logistic Regression and Random Forest models.
- Selects each model's decision threshold from validation data, then evaluates once on the untouched test set.
- Uses PR-AUC alongside ROC-AUC, precision, recall, F1, and confusion matrices. PR-AUC is especially useful with this strongly imbalanced dataset.

Generated charts and `model_summary.csv` are written to `artifacts/`.

## Reproducibility

The script uses a fixed random seed (`42`) for all data splits and stochastic models. It requires Python 3.10 or newer. Results can vary slightly across package versions and hardware, particularly for Random Forest training.

The Kaggle data must not be committed to this repository. Its licensing and distribution terms are managed by Kaggle; download it directly from the source linked above.

## Continuous Integration

GitHub Actions installs the declared dependencies and runs a syntax check on every push and pull request. Full model training is intentionally excluded from CI because it requires the externally hosted dataset.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.

## License

This source code is available under the [MIT License](LICENSE).

## Notes

The notebook supplied with the original project is retained as the historical submission. Its original model scores should not be used because it oversamples before the train/test split, allowing duplicate fraud rows to cross between training and test data.
