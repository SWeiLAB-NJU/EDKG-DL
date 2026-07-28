"""
EDKG-DL element classifier training pipeline.

This script trains and evaluates five binary classification models:

- Random Forest
- Linear Support Vector Classifier
- Decision Tree
- Gaussian Naive Bayes
- K-Nearest Neighbors

Usage
-----
1. Place this script and the six preprocessed CSV files in the same folder.
2. Modify DATA_PREFIX below.
3. Run:

    python edkgdl_element_classifier.py

Required input files
--------------------
{DATA_PREFIX}-Xtrain.csv
{DATA_PREFIX}-Xtrain_.csv
{DATA_PREFIX}-Xtest.csv
{DATA_PREFIX}-Ytrain.csv
{DATA_PREFIX}-Ytrain_.csv
{DATA_PREFIX}-Ytest.csv

Required packages
-----------------
pandas
numpy
joblib
scikit-learn
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# ===================== BASIC CONFIG (EDIT THIS ONLY) =====================
# File prefix only. Do not include ".csv".
DATA_PREFIX = "event_7"
# ========================================================================

# All input and output paths are relative to this script's location.
BASE_DIR = Path(__file__).resolve().parent


def input_path(suffix: str) -> Path:
    """Build the path of an input CSV file."""
    return BASE_DIR / f"{DATA_PREFIX}{suffix}"


def read_feature_csv(path: Path) -> pd.DataFrame:
    """
    Read a feature CSV saved with its first column as the DataFrame index.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path, index_col=0)

    if data.empty:
        raise ValueError(f"Feature file is empty: {path}")

    return data


def read_label_csv(path: Path) -> pd.Series:
    """
    Read a one-dimensional label CSV robustly.

    This handles files saved either with or without a pandas index column.
    Columns such as "Unnamed: 0" are removed before selecting the label.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path)

    unnamed_columns = [
        column
        for column in data.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        data = data.drop(columns=unnamed_columns)

    if data.shape[1] == 0:
        raise ValueError(f"No label column was found in: {path}")

    if data.shape[1] > 1:
        print(
            f"Warning: {path.name} contains multiple non-index columns. "
            f"The first column, {data.columns[0]!r}, will be used as the label."
        )

    labels = data.iloc[:, 0]
    labels.name = "label"
    return labels


def get_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    """
    Return continuous model scores for ROC-AUC and PR-AUC.

    Priority:
    1. Positive-class probability from predict_proba
    2. decision_function score
    3. Predicted class labels as a fallback
    """
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(x)
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                return probabilities[:, 1]
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(x)
            if scores.ndim == 2 and scores.shape[1] == 2:
                return scores[:, 1]
            return scores
        except Exception:
            pass

    return model.predict(x)


def evaluate_split(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Calculate confusion-matrix metrics, ROC-AUC, and PR-AUC.
    """
    labels_present = np.unique(y_true)
    if len(labels_present) != 2:
        raise ValueError(
            "Binary evaluation requires both classes in the evaluated split. "
            f"Classes found: {labels_present.tolist()}"
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = None
    pr_auc = None

    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            roc_auc = None

        try:
            pr_precision, pr_recall, _ = precision_recall_curve(
                y_true,
                y_scores,
            )
            pr_auc = auc(pr_recall, pr_precision)
        except Exception:
            pr_auc = None

    return {
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
    }


def validate_training_data(
    x_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    cv_splits: int,
) -> None:
    """
    Validate feature/label dimensions and cross-validation feasibility.
    """
    if len(x_train_resampled) != len(y_train_resampled):
        raise ValueError(
            "The resampled feature and label files contain different "
            f"numbers of rows: {len(x_train_resampled)} and "
            f"{len(y_train_resampled)}."
        )

    class_counts = y_train_resampled.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "The training labels must contain exactly two classes. "
            f"Class counts: {class_counts.to_dict()}"
        )

    minimum_class_count = int(class_counts.min())
    if minimum_class_count < cv_splits:
        raise ValueError(
            f"{cv_splits}-fold stratified cross-validation requires at least "
            f"{cv_splits} samples in each class, but the smallest class has "
            f"{minimum_class_count}. Reduce cv_splits in run_model()."
        )


def run_model(
    name: str,
    estimator: Any,
    param_grid: Dict[str, Any],
    x_train_resampled: pd.DataFrame,
    y_train_resampled: pd.Series,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 10,
    scoring: str = "f1",
    n_jobs: int = -1,
    save_dir: Path = BASE_DIR,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Tune one model using the resampled training set, save the fitted model and
    cross-validation table, and evaluate it on the original train/test sets.
    """
    print(f"\n===== Begin Train: {name} =====")

    validate_training_data(
        x_train_resampled,
        y_train_resampled,
        cv_splits,
    )

    stratified_cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=stratified_cv,
        refit=True,
    )
    grid_search.fit(
        x_train_resampled,
        np.asarray(y_train_resampled).ravel(),
    )

    print(
        f"Best {scoring}: {grid_search.best_score_:.6f} "
        f"using {grid_search.best_params_}"
    )

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_path = save_dir / f"cv_results_{name}.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"Saved CV results to: {cv_path}")

    best_model = grid_search.best_estimator_
    model_path = save_dir / f"model_{name}.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to: {model_path}")

    train_predictions = best_model.predict(x_train)
    train_scores = get_scores(best_model, x_train)
    train_metrics = evaluate_split(
        y_train,
        train_predictions,
        train_scores,
    )
    print(f"[{name}] Training metrics: {train_metrics}")

    test_predictions = best_model.predict(x_test)
    test_scores = get_scores(best_model, x_test)
    test_metrics = evaluate_split(
        y_test,
        test_predictions,
        test_scores,
    )
    print(f"[{name}] Test metrics: {test_metrics}")

    return train_metrics, test_metrics


def main() -> None:
    """Run the complete element-classifier training workflow."""

    output_dir = BASE_DIR / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data generated by the preprocessing pipeline
    # ------------------------------------------------------------------
    x_train = read_feature_csv(input_path("-Xtrain.csv"))
    x_train_resampled = read_feature_csv(
        input_path("-Xtrain_.csv")
    )
    x_test = read_feature_csv(input_path("-Xtest.csv"))

    y_train = read_label_csv(input_path("-Ytrain.csv"))
    y_train_resampled = read_label_csv(
        input_path("-Ytrain_.csv")
    )
    y_test = read_label_csv(input_path("-Ytest.csv"))

    print(
        "Shapes:",
        "Xtrain",
        x_train.shape,
        "| Ytrain",
        y_train.shape,
        "| Xtrain_",
        x_train_resampled.shape,
        "| Ytrain_",
        y_train_resampled.shape,
        "| Xtest",
        x_test.shape,
        "| Ytest",
        y_test.shape,
    )

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError(
            "Xtrain and Xtest do not contain the same feature columns."
        )

    if list(x_train.columns) != list(x_train_resampled.columns):
        raise ValueError(
            "Xtrain and Xtrain_ do not contain the same feature columns."
        )

    # Store test-set results for the final summary.
    output_rows = []

    # ------------------------------------------------------------------
    # 2. Random Forest
    # ------------------------------------------------------------------
    random_forest_params = {
        "n_estimators": [100, 200, 300, 500],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 10, 20, 50, 100],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "n_jobs": [-1],
        "random_state": [42],
    }

    _, random_forest_test = run_model(
        name="RandomForest",
        estimator=RandomForestClassifier(),
        param_grid=random_forest_params,
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=output_dir,
    )
    output_rows.append(("RandomForest", random_forest_test))

    # ------------------------------------------------------------------
    # 3. Linear SVC
    # ------------------------------------------------------------------
    # LinearSVC does not provide predict_proba, so decision_function is
    # used for ROC-AUC and PR-AUC.
    linear_svc_params = {
        "loss": ["hinge", "squared_hinge"],
        "C": [2.0, 1.0, 0.5, 0.2, 0.1],
        "tol": [1e-4, 1e-3],
        "max_iter": [1000, 2000, 5000],
    }

    _, linear_svc_test = run_model(
        name="LinearSVC",
        estimator=LinearSVC(dual=True),
        param_grid=linear_svc_params,
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=output_dir,
    )
    output_rows.append(("LinearSVC", linear_svc_test))

    # ------------------------------------------------------------------
    # 4. Decision Tree
    # ------------------------------------------------------------------
    decision_tree_params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 50, 100, 200],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [42],
    }

    _, decision_tree_test = run_model(
        name="DecisionTree",
        estimator=DecisionTreeClassifier(),
        param_grid=decision_tree_params,
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=output_dir,
    )
    output_rows.append(("DecisionTree", decision_tree_test))

    # ------------------------------------------------------------------
    # 5. Gaussian Naive Bayes
    # ------------------------------------------------------------------
    _, gaussian_nb_test = run_model(
        name="GaussianNB",
        estimator=GaussianNB(),
        param_grid={},
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=output_dir,
    )
    output_rows.append(("GaussianNB", gaussian_nb_test))

    # ------------------------------------------------------------------
    # 6. K-Nearest Neighbors
    # ------------------------------------------------------------------
    knn_params = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "weights": ["uniform", "distance"],
        "n_jobs": [-1],
    }

    _, knn_test = run_model(
        name="KNeighbors",
        estimator=KNeighborsClassifier(),
        param_grid=knn_params,
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=output_dir,
    )
    output_rows.append(("KNeighbors", knn_test))

    # ------------------------------------------------------------------
    # 7. Aggregate and save test-set performance
    # ------------------------------------------------------------------
    result_columns = [
        "fn",
        "fp",
        "tn",
        "tp",
        "accuracy",
        "f1-score",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ]

    result_data = []
    result_index = []

    for model_name, metrics in output_rows:
        result_index.append(model_name)
        result_data.append(
            [
                metrics["fn"],
                metrics["fp"],
                metrics["tn"],
                metrics["tp"],
                metrics["accuracy"],
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
                metrics["roc_auc"],
                metrics["pr_auc"],
            ]
        )

    result_df = pd.DataFrame(
        result_data,
        index=result_index,
        columns=result_columns,
    )

    print("\n===== Test Set Summary =====")
    print(result_df)

    summary_path = BASE_DIR / "predictive performance.csv"
    result_df.to_csv(summary_path)
    print(f"\nSaved summary to: {summary_path}")

    if len(result_data) == 5:
        print("success!")


if __name__ == "__main__":
    main()
