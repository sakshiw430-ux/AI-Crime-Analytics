"""
crime_model.py
--------------
Machine Learning core of the Crime Analytics Platform.

Responsibilities:
    - Train a RandomForestClassifier to predict crime category
    - Evaluate the trained model (accuracy, classification report, confusion
      matrix, feature importances)
    - Persist / load the trained model (and its label encoders) using joblib
    - Run predictions on new/unseen data and decode results back to labels
    - Simple crime hotspot detection using KMeans clustering on lat/lon
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "crime_rf_model.joblib")


# --------------------------------------------------------------------------- #
# RESULT CONTAINER
# --------------------------------------------------------------------------- #

@dataclass
class TrainingResult:
    """Bundles everything produced by a training run for easy hand-off to the UI."""
    model: RandomForestClassifier
    encoders: dict
    feature_names: list
    accuracy: float
    f1: float
    report: str
    confusion: np.ndarray
    class_labels: list
    feature_importances: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- #
# TRAINING
# --------------------------------------------------------------------------- #

def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    encoders: dict,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """
    Train a RandomForestClassifier to predict crime category from the given
    encoded feature matrix X and target vector y.

    Splits data into train/test, fits the model, and computes standard
    classification metrics plus feature importances for interpretability.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    target_encoder = encoders.get("__target__")
    class_labels = (
        list(target_encoder.classes_) if target_encoder is not None else sorted(y.unique().tolist())
    )

    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in class_labels],
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_test, y_pred)

    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return TrainingResult(
        model=model,
        encoders=encoders,
        feature_names=feature_names,
        accuracy=accuracy,
        f1=f1,
        report=report,
        confusion=conf_matrix,
        class_labels=[str(c) for c in class_labels],
        feature_importances=importances,
    )


# --------------------------------------------------------------------------- #
# PERSISTENCE (joblib)
# --------------------------------------------------------------------------- #

def save_model(result: TrainingResult, path: str = MODEL_PATH) -> str:
    """
    Persist the trained model bundle (model + encoders + feature names) to
    disk using joblib, so it can be reloaded in a later session without
    retraining. Returns the path the model was saved to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bundle = {
        "model": result.model,
        "encoders": result.encoders,
        "feature_names": result.feature_names,
        "class_labels": result.class_labels,
    }
    joblib.dump(bundle, path)
    return path


def load_model(path: str = MODEL_PATH) -> dict:
    """
    Load a previously saved model bundle from disk.
    Raises FileNotFoundError if no saved model exists at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'. Train a model first."
        )
    return joblib.load(path)


def model_exists(path: str = MODEL_PATH) -> bool:
    """Check whether a trained model has already been saved to disk."""
    return os.path.exists(path)


# --------------------------------------------------------------------------- #
# PREDICTION
# --------------------------------------------------------------------------- #

def predict_crime_category(bundle: dict, X_new: pd.DataFrame) -> pd.Series:
    """
    Run predictions on new data using a loaded model bundle.

    X_new must contain the same feature columns the model was trained on
    (missing columns are filled with 0; extra columns are ignored).
    Returns predicted crime-category labels (decoded, human-readable).
    """
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_names = bundle["feature_names"]

    X_aligned = X_new.copy()
    for col in feature_names:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    X_aligned = X_aligned[feature_names]

    # Encode any object columns using the encoders fitted during training
    for col in X_aligned.columns:
        if col in encoders and X_aligned[col].dtype in ("object", "string"):
            le = encoders[col]
            X_aligned[col] = X_aligned[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            X_aligned[col] = le.transform(X_aligned[col])

    preds_encoded = model.predict(X_aligned)

    target_encoder = encoders.get("__target__")
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(preds_encoded)
    else:
        preds = preds_encoded

    return pd.Series(preds, index=X_new.index, name="Predicted_Crime_Category")


# --------------------------------------------------------------------------- #
# HOTSPOT DETECTION (KMeans clustering on geographic coordinates)
# --------------------------------------------------------------------------- #

def detect_hotspots(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    n_clusters: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect crime hotspots by clustering incident locations (latitude,
    longitude) using KMeans. Adds a 'Hotspot_Cluster' column to the returned
    DataFrame and reports cluster sizes so the largest clusters can be
    highlighted as the most severe hotspots.
    """
    geo_df = df.dropna(subset=[lat_col, lon_col]).copy()
    geo_df[lat_col] = pd.to_numeric(geo_df[lat_col], errors="coerce")
    geo_df[lon_col] = pd.to_numeric(geo_df[lon_col], errors="coerce")
    geo_df = geo_df.dropna(subset=[lat_col, lon_col])

    if geo_df.empty:
        raise ValueError("No valid latitude/longitude values available for hotspot detection.")

    n_clusters = max(1, min(n_clusters, len(geo_df)))
    coords = geo_df[[lat_col, lon_col]].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    geo_df["Hotspot_Cluster"] = kmeans.fit_predict(coords)

    cluster_sizes = geo_df["Hotspot_Cluster"].value_counts().rename("Incident_Count")
    geo_df = geo_df.merge(cluster_sizes, left_on="Hotspot_Cluster", right_index=True)

    # Rank clusters by severity (0 = most severe / largest cluster)
    rank_map = {
        cluster_id: rank
        for rank, cluster_id in enumerate(cluster_sizes.sort_values(ascending=False).index)
    }
    geo_df["Hotspot_Rank"] = geo_df["Hotspot_Cluster"].map(rank_map)

    return geo_df