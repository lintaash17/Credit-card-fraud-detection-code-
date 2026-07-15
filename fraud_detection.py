"""Place creditcard.csv in this directory, then run:
    python fraud_detection.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("creditcard.csv")
ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE = 42


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose a threshold on validation data only, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision_recall_curve returns one more precision/recall value than thresholds.
    best_index = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_index])


def evaluate(
    name: str,
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    threshold = choose_f1_threshold(y_validation, validation_probabilities)
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    print(f"\n{name}")
    print(f"Validation PR-AUC: {average_precision_score(y_validation, validation_probabilities):.4f}")
    print(f"Test ROC-AUC:      {roc_auc_score(y_test, test_probabilities):.4f}")
    print(f"Test PR-AUC:       {average_precision_score(y_test, test_probabilities):.4f}")
    print(f"Selected threshold: {threshold:.4f}")
    print(classification_report(y_test, test_predictions, digits=4, zero_division=0))

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, ax=axes[0], colorbar=False)
    axes[0].set_title(f"{name}: test confusion matrix")
    PrecisionRecallDisplay.from_predictions(y_test, test_probabilities, ax=axes[1], name=name)
    axes[1].set_title(f"{name}: test precision-recall curve")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / f"{name.lower().replace(' ', '_')}_evaluation.png", dpi=160)
    plt.close(figure)

    return {
        "name": name,
        "threshold": threshold,
        "validation_pr_auc": average_precision_score(y_validation, validation_probabilities),
        "test_pr_auc": average_precision_score(y_test, test_probabilities),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Download it from Kaggle and place it beside fraud_detection.py."
        )

    data = pd.read_csv(DATA_PATH)
    expected_columns = {"Time", "Amount", "Class"}
    if not expected_columns.issubset(data.columns):
        raise ValueError("The CSV must contain Time, Amount, Class, and the anonymized V1-V28 features.")
    if data["Class"].nunique() != 2:
        raise ValueError("Class must contain both 0 (legitimate) and 1 (fraud) labels.")

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(data):,} rows; removed {duplicate_count:,} exact duplicates.")
    print("Class distribution:\n", data["Class"].value_counts())

    X = data.drop(columns="Class")
    y = data["Class"]

    # The test set stays untouched and retains the real fraud prevalence.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development, y_development, test_size=0.25, stratify=y_development, random_state=RANDOM_STATE
    )

    numeric_features = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        [("scale_time_amount", StandardScaler(), numeric_features)], remainder="passthrough"
    )
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    for name, classifier in models.items():
        model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_validation, y_validation, X_test, y_test))

    summary = pd.DataFrame(results).sort_values("validation_pr_auc", ascending=False)
    summary.to_csv(ARTIFACTS_DIR / "model_summary.csv", index=False)
    print("\nModel comparison (ranked by validation PR-AUC):")
    print(summary.to_string(index=False))
    print(f"\nSaved charts and summary to: {ARTIFACTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
