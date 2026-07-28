"""
EDKG-DL data preprocessing pipeline.

Usage
-----
1. Place this script and the input CSV file in the same folder.
2. Modify TARGET_FILE below without adding the ".csv" suffix.
3. Run in VS Code or from a terminal:

    python edkgdl_data_pipeline.py

Required packages
-----------------
pandas
scikit-learn
imbalanced-learn
"""

from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


# ===================== Basic Config (only modify this) =====================
# Do not include the .csv suffix.
TARGET_FILE = "event_7-abortion_finger"  # Change this to your file prefix
# ==========================================================================

# Input and output files are located in the same folder as this Python script.
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / f"{TARGET_FILE}.csv"


def output_path(suffix: str) -> Path:
    """Build an output path in the script directory."""
    return BASE_DIR / f"{TARGET_FILE}{suffix}"


def main() -> None:
    """Run the complete data preprocessing pipeline."""

    # ----------------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------------
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {CSV_PATH}\n"
            "Please place the CSV file in the same folder as this script "
            "and check TARGET_FILE."
        )

    data = pd.read_csv(CSV_PATH)
    print("Input file:", CSV_PATH)
    print("Raw shape:", data.shape)

    if data.shape[1] < 3:
        raise ValueError(
            "The input file must contain at least three columns: "
            "an ID column, one feature column, and one label column."
        )

    # Count active/inactive samples; the last column is treated as the label.
    num_active = (data.iloc[:, -1] != 0).sum()
    num_inactive = (data.iloc[:, -1] == 0).sum()
    print("Number of active:", num_active)
    print("Number of inactive:", num_inactive)

    # ----------------------------------------------------------------------
    # 2. Clean columns
    # ----------------------------------------------------------------------
    # Drop columns containing the literal string "#NAME?".
    mask_bad = data.apply(lambda column: (column == "#NAME?").any())
    bad_cols = data.columns[mask_bad].tolist()
    if bad_cols:
        print("Drop columns containing #NAME?:", bad_cols)
        data = data.drop(columns=bad_cols)

    # Convert values to numeric; non-convertible values become NaN.
    data_num = data.apply(pd.to_numeric, errors="coerce")

    # Drop columns containing only NaN.
    all_nan_cols = [
        column for column in data_num.columns
        if data_num[column].isna().all()
    ]
    if all_nan_cols:
        print("Drop all-NaN columns:", all_nan_cols)
        data_num = data_num.drop(columns=all_nan_cols)

    # Drop columns containing any NaN.
    na_cols = data_num.columns[data_num.isna().any()].tolist()
    if na_cols:
        print("Drop columns containing NaN:", na_cols)
        data_num = data_num.drop(columns=na_cols)

    if data_num.shape[1] < 3:
        raise ValueError(
            "Fewer than three usable columns remain after data cleaning."
        )

    # The first column is treated as an ID and the last column as the label.
    # If the first column is not an ID, change `1:-1` to `:-1`.
    x = data_num.iloc[:, 1:-1].copy()
    y = data_num.iloc[:, -1].copy()

    if x.empty:
        raise ValueError("No feature columns remain after data cleaning.")

    print("X shape:", x.shape)
    print("y shape:", y.shape)
    print("y unique:", y.unique())

    # Drop features whose maximum value is at least 10,000.
    extreme_cols = [
        column for column in x.columns
        if x[column].max() >= 10000
    ]
    if extreme_cols:
        print("Drop extreme columns (>=10000):", extreme_cols)
        x = x.drop(columns=extreme_cols)

    if x.empty:
        raise ValueError("No feature columns remain after extreme-value filtering.")

    print("X shape after extreme filter:", x.shape)

    # ----------------------------------------------------------------------
    # 3. Remove zero-variance features
    # ----------------------------------------------------------------------
    variance_selector = VarianceThreshold()
    x_var0_array = variance_selector.fit_transform(x)
    variance_features = x.columns[
        variance_selector.get_support()
    ].tolist()

    x_var0 = pd.DataFrame(
        x_var0_array,
        columns=variance_features,
        index=x.index,
    )
    print("After VarianceThreshold:", x_var0.shape)

    if x_var0.empty:
        raise ValueError("No features remain after variance filtering.")

    # ----------------------------------------------------------------------
    # 4. Mutual-information feature selection
    # ----------------------------------------------------------------------
    mi_scores = mutual_info_classif(
        x_var0,
        y,
        random_state=0,
    )
    k = max(int((mi_scores > 0).sum()), 1)

    mi_selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k,
    )
    x_fsmic_array = mi_selector.fit_transform(x_var0, y)

    selected_features = x_var0.columns[
        mi_selector.get_support()
    ].tolist()
    x_fsmic = pd.DataFrame(
        x_fsmic_array,
        columns=selected_features,
        index=x_var0.index,
    )

    # Cross-validation using Random Forest.
    cv_score = cross_val_score(
        RandomForestClassifier(
            n_estimators=10,
            random_state=0,
        ),
        x_fsmic,
        y,
        cv=5,
    ).mean()

    print("k (MI > 0):", k)
    print("CV mean score:", cv_score)
    print("X_fsmic shape:", x_fsmic.shape)

    # ----------------------------------------------------------------------
    # 5. Save selected features and labels
    # ----------------------------------------------------------------------
    x_selected_path = output_path("-2.csv")
    y_path = output_path("-3.csv")

    x_fsmic.to_csv(x_selected_path)
    y.to_csv(y_path)

    # Keep the same read-back behavior as the original notebook.
    x_selected = pd.read_csv(x_selected_path, index_col=0)
    y_selected = pd.read_csv(y_path, index_col=0).values.ravel()

    print("X_ shape:", x_selected.shape, "| y_ shape:", y_selected.shape)

    # ----------------------------------------------------------------------
    # 6. Standardization and train/test split
    # ----------------------------------------------------------------------
    scaler = StandardScaler()
    x_std = pd.DataFrame(
        scaler.fit_transform(x_selected),
        columns=x_selected.columns,
        index=x_selected.index,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_std,
        y_selected,
        test_size=0.2,
        random_state=0,
        stratify=y_selected,
    )

    print(
        "Xtrain/Xtest/Ytrain/Ytest shapes:",
        x_train.shape,
        x_test.shape,
        (len(y_train),),
        (len(y_test),),
    )

    y_train = pd.Series(y_train, name="label")
    y_test = pd.Series(y_test, name="label")

    print("Train class counts:\n", y_train.value_counts())
    print("Test class counts:\n", y_test.value_counts())

    # ----------------------------------------------------------------------
    # 7. Apply SMOTE only to the training set
    # ----------------------------------------------------------------------
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(
        x_train,
        y_train,
    )

    # Restore clear column/index metadata after SMOTE.
    x_train_resampled = pd.DataFrame(
        x_train_resampled,
        columns=x_train.columns,
    )
    y_train_resampled = pd.Series(
        y_train_resampled,
        name="label",
    )

    print("After SMOTE n_samples:", x_train_resampled.shape[0])
    print(
        "Resampled train class counts:\n",
        y_train_resampled.value_counts(),
    )

    # ----------------------------------------------------------------------
    # 8. Save data splits
    # ----------------------------------------------------------------------
    output_files = {
        "-Xtrain.csv": x_train,
        "-Xtrain_.csv": x_train_resampled,
        "-Xtest.csv": x_test,
        "-Ytrain.csv": y_train,
        "-Ytrain_.csv": y_train_resampled,
        "-Ytest.csv": y_test,
    }

    for suffix, dataframe in output_files.items():
        dataframe.to_csv(output_path(suffix))

    print("\nGenerated files:")
    print(" -", x_selected_path.name)
    print(" -", y_path.name)
    for suffix in output_files:
        print(" -", output_path(suffix).name)

    print("\nAll done ✔")


if __name__ == "__main__":
    main()
