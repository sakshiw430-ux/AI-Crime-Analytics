"""
preprocessing.py
-----------------
Data loading and preprocessing utilities for the Crime Analytics Platform.

Responsibilities:
    - Load an uploaded CSV into a pandas DataFrame safely
    - Clean the data (handle missing values, duplicates, whitespace)
    - Parse date/time columns and engineer time-based features
    - Encode categorical columns for machine learning
    - Build the final feature matrix (X) / target vector (y) used to train
      the Random Forest model in crime_model.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import detect_all_columns


# --------------------------------------------------------------------------- #
# LOADING
# --------------------------------------------------------------------------- #

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file (a Streamlit UploadedFile or a path) into a DataFrame.

    Raises a ValueError with a friendly message if the file cannot be parsed,
    so the caller (app.py) can surface it cleanly in the UI.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the UI
        raise ValueError(f"Could not read the uploaded CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    # Normalize column names: strip whitespace only (keep original casing so
    # detect_column's fuzzy matching / user-facing labels still look natural)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# CLEANING
# --------------------------------------------------------------------------- #

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform general-purpose cleaning on a raw crime dataset:
        - drop fully-empty rows/columns
        - drop exact duplicate rows
        - strip whitespace from text/object columns
        - fill missing categorical values with 'Unknown'
        - fill missing numeric values with the column median
    """
    df = df.copy()

    # Drop rows/columns that are entirely empty
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Drop exact duplicate records
    df = df.drop_duplicates()

    # Strip whitespace on text columns (covers both classic 'object' dtype
    # and the newer pandas 'string' dtype, e.g. pandas >= 2.x/3.x backends)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Fill missing values: text/categorical columns get 'Unknown',
    # numeric columns get filled with the column median.
    numeric_cols = df.select_dtypes(include="number").columns
    for col in df.columns:
        if col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna("Unknown")

    df = df.reset_index(drop=True)
    return df


def engineer_datetime_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """
    Parse a detected date/datetime column and engineer useful time-based
    features: Year, Month, Day, Hour, DayOfWeek. Non-parsable dates become
    NaT and downstream NaNs are safely filled with -1.

    If no date column is available, the DataFrame is returned unchanged.
    """
    df = df.copy()
    if not date_col or date_col not in df.columns:
        return df

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    df["Crime_Year"] = parsed.dt.year
    df["Crime_Month"] = parsed.dt.month
    df["Crime_Day"] = parsed.dt.day
    df["Crime_Hour"] = parsed.dt.hour
    df["Crime_DayOfWeek"] = parsed.dt.dayofweek  # 0=Monday

    time_cols = ["Crime_Year", "Crime_Month", "Crime_Day", "Crime_Hour", "Crime_DayOfWeek"]
    df[time_cols] = df[time_cols].fillna(-1).astype(int)
    return df


def full_preprocessing_pipeline(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Run the complete preprocessing pipeline used throughout the app:
        1. clean_data
        2. detect semantic columns (crime type, date, district, lat/lon)
        3. engineer datetime features

    Returns the processed DataFrame plus a dict of detected column names,
    so downstream modules (visualization, modeling) don't need to re-detect.
    """
    df = clean_data(raw_df)
    columns_map = detect_all_columns(df)
    df = engineer_datetime_features(df, columns_map.get("date"))
    return df, columns_map


# --------------------------------------------------------------------------- #
# ENCODING / FEATURE PREPARATION FOR ML
# --------------------------------------------------------------------------- #

def encode_categoricals(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode the given categorical columns in place (on a copy).
    Returns the encoded DataFrame and a dict of fitted LabelEncoders keyed
    by column name, so predictions on new data can reuse the same mapping.
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def build_feature_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series, list, dict]:
    """
    Build the (X, y) feature/target split used to train the Random Forest
    crime-category classifier.

    - Numeric columns are used as-is.
    - Object (categorical) columns other than the target are label-encoded.
    - The target column itself is label-encoded separately.
    - Columns in `exclude_cols` (e.g. free-text description, raw date string,
      IDs) are dropped from the feature set.

    Returns:
        X: encoded feature DataFrame
        y: encoded target Series
        feature_names: list of column names used as features
        encoders: dict of {column_name: LabelEncoder}, including the target
                  under the key "__target__"
    """
    exclude_cols = exclude_cols or []
    working = df.copy()

    # Drop rows where the target itself is missing
    working = working.dropna(subset=[target_col])

    feature_df = working.drop(columns=[c for c in exclude_cols if c in working.columns], errors="ignore")
    feature_df = feature_df.drop(columns=[target_col], errors="ignore")

    # Only keep reasonably usable columns (drop very high-cardinality text
    # columns like free-text descriptions/IDs which hurt a plain RF model)
    text_like_cols = set(feature_df.select_dtypes(include=["object", "string"]).columns)
    usable_cols = []
    for col in feature_df.columns:
        if col in text_like_cols:
            if feature_df[col].nunique() <= max(50, int(0.5 * len(feature_df))):
                usable_cols.append(col)
        else:
            usable_cols.append(col)
    feature_df = feature_df[usable_cols]

    categorical_cols = list(feature_df.select_dtypes(include=["object", "string"]).columns)
    feature_df, encoders = encode_categoricals(feature_df, categorical_cols)

    target_encoder = LabelEncoder()
    y = pd.Series(target_encoder.fit_transform(working[target_col].astype(str)), index=working.index, name=target_col)
    encoders["__target__"] = target_encoder

    X = feature_df
    feature_names = list(X.columns)
    return X, y, feature_names, encoders