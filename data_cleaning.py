import pandas as pd
import numpy as np


def validate_dataset(df, target):
    """
    Performs basic dataset validation.
    """

    validation = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isnull().sum().sum()),
        "target_exists": target in df.columns,
        "target_missing": 0
    }

    if target in df.columns:
        validation["target_missing"] = int(df[target].isnull().sum())

    return validation


def clean_dataset(df, target):
    """
    Automatically cleans the dataset.
    """

    df = df.copy()

    cleaning_log = []

    # -------------------------------------------------
    # 1. Remove completely empty columns
    # -------------------------------------------------

    empty_columns = [
        col for col in df.columns
        if df[col].isnull().all()
    ]

    if empty_columns:
        df.drop(columns=empty_columns, inplace=True)

        cleaning_log.append(
            f"Removed completely empty columns: {empty_columns}"
        )

    # -------------------------------------------------
    # 2. Convert target to numeric if possible
    # -------------------------------------------------

    if target not in df.columns:
        raise ValueError("Selected target column does not exist.")

    if not pd.api.types.is_numeric_dtype(df[target]):

        converted_target = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        valid_ratio = converted_target.notna().mean()

        if valid_ratio >= 0.90:
            df[target] = converted_target

            cleaning_log.append(
                f"Converted target column '{target}' to numeric."
            )

        else:
            raise ValueError(
                f"Target column '{target}' must be numeric "
                f"for regression."
            )

    # -------------------------------------------------
    # 3. Remove rows with missing target
    # -------------------------------------------------

    before = len(df)

    df.dropna(subset=[target], inplace=True)

    removed = before - len(df)

    if removed > 0:
        cleaning_log.append(
            f"Removed {removed} rows with missing target values."
        )

    # -------------------------------------------------
    # 4. Remove duplicate rows
    # -------------------------------------------------

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        df.drop_duplicates(inplace=True)

        cleaning_log.append(
            f"Removed {duplicates} duplicate rows."
        )

    # -------------------------------------------------
    # 5. Remove constant columns
    # -------------------------------------------------

    constant_columns = []

    for col in df.columns:

        if col == target:
            continue

        if df[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

    if constant_columns:

        df.drop(columns=constant_columns, inplace=True)

        cleaning_log.append(
            f"Removed constant columns: {constant_columns}"
        )

    # -------------------------------------------------
    # 6. Fill missing numeric values
    # -------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_columns:

        if col == target:
            continue

        missing = df[col].isnull().sum()

        if missing > 0:

            median_value = df[col].median()

            df[col] = df[col].fillna(median_value)

            cleaning_log.append(
                f"Filled {missing} missing values in "
                f"'{col}' using median."
            )

    # -------------------------------------------------
    # 7. Fill missing categorical values
    # -------------------------------------------------

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    for col in categorical_columns:

        missing = df[col].isnull().sum()

        if missing > 0:

            mode_values = df[col].mode()

            if len(mode_values) > 0:

                df[col] = df[col].fillna(
                    mode_values.iloc[0]
                )

                cleaning_log.append(
                    f"Filled {missing} missing values in "
                    f"'{col}' using mode."
                )

    # -------------------------------------------------
    # Final validation
    # -------------------------------------------------

    if len(df) < 5:
        raise ValueError(
            "Dataset has too few rows after cleaning."
        )

    if df[target].nunique() <= 1:
        raise ValueError(
            "Target column has only one unique value. "
            "Regression cannot be performed."
        )

    return df, cleaning_log