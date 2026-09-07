"""
EDKG-DL element regressor training pipeline.

This script trains and evaluates five regression models:

- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor

"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "The xgboost package is required. Install it with:\n"
        "pip install xgboost"
    ) from exc

DATA_PREFIX = "../quantitative_data_for_modeling/Event_47"

RANDOM_STATE = 42
CV_SPLITS = 10
SCORING = "neg_mean_squared_error"
N_JOBS = -1

# Input and output paths are relative to this Python script.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "regression_outputs"


def data_path(suffix: str) -> Path:
    """Build an input-file path in the script directory."""
    return BASE_DIR / f"{DATA_PREFIX}{suffix}"


def read_feature_csv(path: Path) -> pd.DataFrame:
    """
    Read a feature CSV whose first column was saved as a pandas index.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path, index_col=0)

    if data.empty:
        raise ValueError(f"Feature file is empty: {path}")

    data = data.apply(pd.to_numeric, errors="raise")
    return data


def read_target_csv(path: Path) -> pd.Series:
    """
    Read a one-dimensional regression target robustly.

    Files saved by pandas may contain an extra index column named
    "Unnamed: 0". Such columns are removed before selecting the target.
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
        raise ValueError(f"No target column was found in: {path}")

    if data.shape[1] > 1:
        print(
            f"Warning: {path.name} contains multiple non-index columns. "
            f"The first column, {data.columns[0]!r}, will be used as the target."
        )

    target = pd.to_numeric(data.iloc[:, 0], errors="raise")
    target.name = "target"
    return target


def read_optional_shape(
    feature_path: Path,
    target_path: Path,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int]]]:
    """
    Read optional resampled files only to report their dimensions.

    Resampled data are not used to fit the regression models.
    """
    feature_shape = None
    target_shape = None

    if feature_path.exists():
        feature_shape = read_feature_csv(feature_path).shape

    if target_path.exists():
        target_shape = read_target_csv(target_path).shape

    return feature_shape, target_shape


def validate_datasets(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int,
) -> None:
    """Validate dimensions, columns, numeric values, and CV feasibility."""
    if len(x_train) != len(y_train):
        raise ValueError(
            "Xtrain and Ytrain contain different numbers of rows: "
            f"{len(x_train)} and {len(y_train)}."
        )

    if len(x_test) != len(y_test):
        raise ValueError(
            "Xtest and Ytest contain different numbers of rows: "
            f"{len(x_test)} and {len(y_test)}."
        )

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError(
            "Xtrain and Xtest do not contain the same feature columns "
            "in the same order."
        )

    if len(x_train) < cv_splits:
        raise ValueError(
            f"{cv_splits}-fold cross-validation requires at least "
            f"{cv_splits} training samples, but only {len(x_train)} were found."
        )

    arrays_to_check = {
        "Xtrain": x_train.to_numpy(),
        "Ytrain": y_train.to_numpy(),
        "Xtest": x_test.to_numpy(),
        "Ytest": y_test.to_numpy(),
    }

    for name, values in arrays_to_check.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )


def can_compute_msle(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> bool:
    """Return whether MSLE can be calculated for the supplied values."""
    return bool(
        np.min(np.asarray(y_true)) >= 0
        and np.min(np.asarray(y_pred)) >= 0
    )


def evaluate_split(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Calculate common regression metrics for one dataset split."""
    metrics: Dict[str, Optional[float]] = {
        "max_error": float(max_error(y_true, y_pred)),
        "mean_absolute_error": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "mean_squared_error": float(
            mean_squared_error(y_true, y_pred)
        ),
        "mean_squared_log_error": None,
        "r2_score": float(r2_score(y_true, y_pred)),
    }

    if can_compute_msle(y_true, y_pred):
        try:
            metrics["mean_squared_log_error"] = float(
                mean_squared_log_error(y_true, y_pred)
            )
        except ValueError:
            metrics["mean_squared_log_error"] = None

    return metrics


def parity_plot(
    y_true_train: pd.Series,
    y_pred_train: np.ndarray,
    y_true_test: pd.Series,
    y_pred_test: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Create and save train/test measured-versus-predicted plots."""
    fig, (ax_train, ax_test) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
    )

    train_mae = mean_absolute_error(
        y_true_train,
        y_pred_train,
    )
    ax_train.scatter(
        y_pred_train,
        y_true_train,
        s=30,
        alpha=0.35,
        marker="*",
        label=f"MAE={train_mae:.3f}",
    )
    train_low = min(
        np.min(y_true_train),
        np.min(y_pred_train),
    )
    train_high = max(
        np.max(y_true_train),
        np.max(y_pred_train),
    )
    ax_train.plot(
        [train_low, train_high],
        [train_low, train_high],
        linestyle=":",
        linewidth=1,
    )
    ax_train.set_title("Training set", fontsize=13)
    ax_train.set_xlabel("Predicted", fontsize=12)
    ax_train.set_ylabel("Measured", fontsize=12)
    ax_train.legend()

    test_mae = mean_absolute_error(
        y_true_test,
        y_pred_test,
    )
    ax_test.scatter(
        y_pred_test,
        y_true_test,
        s=30,
        alpha=0.35,
        marker="o",
        label=f"MAE={test_mae:.3f}",
    )
    test_low = min(
        np.min(y_true_test),
        np.min(y_pred_test),
    )
    test_high = max(
        np.max(y_true_test),
        np.max(y_pred_test),
    )
    ax_test.plot(
        [test_low, test_high],
        [test_low, test_high],
        linestyle=":",
        linewidth=1,
    )
    ax_test.set_title("Test set", fontsize=13)
    ax_test.set_xlabel("Predicted", fontsize=12)
    ax_test.set_ylabel("Measured", fontsize=12)
    ax_test.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_model(
    name: str,
    estimator: Any,
    param_grid: Dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path,
    scoring: str = SCORING,
    cv_splits: int = CV_SPLITS,
    n_jobs: int = N_JOBS,
) -> Tuple[
    Dict[str, Optional[float]],
    Dict[str, Optional[float]],
]:
    """
    Tune, fit, save, and evaluate one regression model.
    """
    print(f"\n===== Begin Train: {name} =====")

    cross_validation = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cross_validation,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(
        x_train,
        np.asarray(y_train).ravel(),
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
    test_predictions = best_model.predict(x_test)

    train_metrics = evaluate_split(
        y_train,
        train_predictions,
    )
    test_metrics = evaluate_split(
        y_test,
        test_predictions,
    )

    print(f"[{name}] Training metrics: {train_metrics}")
    print(f"[{name}] Test metrics: {test_metrics}")

    plot_path = save_dir / f"parity_{name}.png"
    parity_plot(
        y_true_train=y_train,
        y_pred_train=train_predictions,
        y_true_test=y_test,
        y_pred_test=test_predictions,
        title=name,
        save_path=plot_path,
    )
    print(f"Saved parity plot to: {plot_path}")

    return train_metrics, test_metrics


def main() -> None:
    """Run the complete element-regressor training workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the original training and test sets
    # ------------------------------------------------------------------
    x_train = read_feature_csv(data_path("-Xtrain.csv"))
    x_test = read_feature_csv(data_path("-Xtest.csv"))
    y_train = read_target_csv(data_path("-Ytrain.csv"))
    y_test = read_target_csv(data_path("-Ytest.csv"))

    # These files are produced by the preprocessing pipeline but are not
    # appropriate for the regression models below. They are loaded only to
    # report their dimensions when available.
    x_train_resampled_shape, y_train_resampled_shape = read_optional_shape(
        data_path("-Xtrain_.csv"),
        data_path("-Ytrain_.csv"),
    )

    validate_datasets(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cv_splits=CV_SPLITS,
    )

    print(
        "Shapes:",
        "Xtrain",
        x_train.shape,
        "| Ytrain",
        y_train.shape,
        "| Xtrain_",
        x_train_resampled_shape,
        "| Ytrain_",
        y_train_resampled_shape,
        "| Xtest",
        x_test.shape,
        "| Ytest",
        y_test.shape,
    )

    test_results = []

    # ------------------------------------------------------------------
    # 2. Random Forest Regressor
    # ------------------------------------------------------------------
    random_forest_params = {
        "n_estimators": [200, 300, 500],
        "criterion": [
            "squared_error",
            "absolute_error",
            "friedman_mse",
        ],
        "max_depth": [None, 10, 20, 50, 100],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [RANDOM_STATE],
        "n_jobs": [-1],
    }

    _, random_forest_test = run_model(
        name="RandomForestRegressor",
        estimator=RandomForestRegressor(),
        param_grid=random_forest_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=OUTPUT_DIR,
    )
    test_results.append(
        ("RandomForestRegressor", random_forest_test)
    )

    # ------------------------------------------------------------------
    # 3. Decision Tree Regressor
    # ------------------------------------------------------------------
    decision_tree_params = {
        "criterion": [
            "squared_error",
            "absolute_error",
            "friedman_mse",
        ],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 50, 100, 200],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [RANDOM_STATE],
    }

    _, decision_tree_test = run_model(
        name="DecisionTreeRegressor",
        estimator=DecisionTreeRegressor(),
        param_grid=decision_tree_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=OUTPUT_DIR,
    )
    test_results.append(
        ("DecisionTreeRegressor", decision_tree_test)
    )

    # ------------------------------------------------------------------
    # 4. XGBoost Regressor
    # ------------------------------------------------------------------
    xgboost_params = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "random_state": [RANDOM_STATE],
        "tree_method": ["hist"],
    }

    _, xgboost_test = run_model(
        name="XGBRegressor",
        estimator=xgb.XGBRegressor(
            objective="reg:squarederror",
        ),
        param_grid=xgboost_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=OUTPUT_DIR,
    )
    test_results.append(
        ("XGBRegressor", xgboost_test)
    )

    # ------------------------------------------------------------------
    # 5. K-Nearest Neighbors Regressor
    # ------------------------------------------------------------------
    knn_params = {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "algorithm": [
            "auto",
            "ball_tree",
            "kd_tree",
            "brute",
        ],
        "weights": ["uniform", "distance"],
        "n_jobs": [-1],
    }

    _, knn_test = run_model(
        name="KNeighborsRegressor",
        estimator=KNeighborsRegressor(),
        param_grid=knn_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=OUTPUT_DIR,
    )
    test_results.append(
        ("KNeighborsRegressor", knn_test)
    )

    # ------------------------------------------------------------------
    # 6. Support Vector Regressor
    # ------------------------------------------------------------------
    svr_params = {
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "gamma": ["scale", "auto"],
        "degree": [2, 3],
        "epsilon": [0.1, 0.2, 0.5],
        "cache_size": [5000],
        "max_iter": [-1],
    }

    _, svr_test = run_model(
        name="SVR",
        estimator=SVR(),
        param_grid=svr_params,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        save_dir=OUTPUT_DIR,
    )
    test_results.append(("SVR", svr_test))

    # ------------------------------------------------------------------
    # 7. Aggregate and save test-set performance
    # ------------------------------------------------------------------
    result_columns = [
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "r2_score",
    ]

    result_rows = []
    result_index = []

    for model_name, metrics in test_results:
        result_index.append(model_name)
        result_rows.append(
            [
                metrics["max_error"],
                metrics["mean_absolute_error"],
                metrics["mean_squared_error"],
                metrics["mean_squared_log_error"],
                metrics["r2_score"],
            ]
        )

    result_df = pd.DataFrame(
        result_rows,
        index=result_index,
        columns=result_columns,
    )

    print("\n===== Test Set Summary (Regression) =====")
    print(result_df)

    summary_path = BASE_DIR / "predictive performance (regression).csv"
    result_df.to_csv(summary_path)
    print(f"\nSaved summary to: {summary_path}")

    if len(test_results) == 5:
        print("success!")


if __name__ == "__main__":
    main()
