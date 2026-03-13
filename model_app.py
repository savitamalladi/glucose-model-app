# model_app.py
"""
Train a realistic "logbook + history" model to predict glucose 1 hour ahead (bg+1:00)
using the last 3 hours of history (bg/carbs/insulin/steps).

Run:
    python model_app.py

Output:
    models/app_history_model.joblib
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "bg+1:00"



# Helpers to locate time-lag columns

def find_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return all columns like 'prefix-<hh>:<mm>' e.g., bg-0:05."""
    return [c for c in df.columns if c.startswith(prefix + "-")]


def parse_minutes(colname: str) -> int:
    """Convert 'bg-5:55' -> 355 minutes."""
    suffix = colname.split("-", 1)[1]
    hh, mm = suffix.split(":")
    return int(hh) * 60 + int(mm)


def most_recent_cols(df: pd.DataFrame, prefix: str, k: int) -> list[str]:
    """
    Pick the k most-recent columns (closest-to-now).
    'Most recent' = smallest minutes in the suffix.
    """
    cols = find_cols(df, prefix)
    if not cols:
        return []
    cols_sorted = sorted(cols, key=parse_minutes)  # smallest minutes first (closest to now)
    return cols_sorted[: min(k, len(cols_sorted))]



# Feature builder (3 hours history)

def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build compact, realistic features for a logbook app.
    Uses last 3 hours (36 x 5-min points) for each signal if available.
    """
    # 3 hours = 180 minutes; if data is 5-min cadence -> 36 points
    bg_cols = most_recent_cols(df, "bg", k=36)
    carbs_cols = most_recent_cols(df, "carbs", k=36)
    insulin_cols = most_recent_cols(df, "insulin", k=36)
    steps_cols = most_recent_cols(df, "steps", k=36)

    if not bg_cols:
        raise ValueError(
            "No 'bg-*' columns found in train.csv. "
            "Check your column names (expected prefixes: bg-, carbs-, insulin-, steps-)."
        )

    # Convert column groups to numeric safely (DataFrame-wide)
    bg_mat = df[bg_cols].apply(pd.to_numeric, errors="coerce")

    carbs_mat = df[carbs_cols].apply(pd.to_numeric, errors="coerce") if carbs_cols else None
    insulin_mat = df[insulin_cols].apply(pd.to_numeric, errors="coerce") if insulin_cols else None
    steps_mat = df[steps_cols].apply(pd.to_numeric, errors="coerce") if steps_cols else None

    X = pd.DataFrame(index=df.index)

    # Current bg (most recent column)
    X["bg_now"] = bg_mat.iloc[:, 0]

    # Stats over last 3 hours
    X["bg_mean_3h"] = bg_mat.mean(axis=1)
    X["bg_std_3h"] = bg_mat.std(axis=1)

    # Totals over last 3 hours
    X["carbs_sum_3h"] = carbs_mat.sum(axis=1) if carbs_mat is not None else 0.0
    X["insulin_sum_3h"] = insulin_mat.sum(axis=1) if insulin_mat is not None else 0.0
    X["steps_sum_3h"] = steps_mat.sum(axis=1) if steps_mat is not None else 0.0

    # Trend over 60 minutes: now - 60 mins ago (12 * 5min)
    # If fewer columns exist, fall back to 0 trend.
    if bg_mat.shape[1] >= 13:
        X["bg_trend_60m"] = bg_mat.iloc[:, 0] - bg_mat.iloc[:, 12]
    else:
        X["bg_trend_60m"] = 0.0

    return X


# Train + save model

def main() -> None:
    # Load data
    train = pd.read_csv("train.csv", low_memory=False)

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train.csv")

    # Build features + target
    y = pd.to_numeric(train[TARGET], errors="coerce")
    X = build_history_features(train)

    # Drop rows where target is missing
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Pipeline: fill missing values -> train model
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("hgb", HistGradientBoostingRegressor(
            random_state=42,
            max_depth=6,
            learning_rate=0.08,
            max_iter=500
        ))
    ])

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit + evaluate
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)

    print("Validation MAE (3-hour logbook model):", mae)
    print("Features used:", list(X.columns))

    # Save trained model
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "app_history_model.joblib")
    joblib.dump(model, out_path)
    print(f"✅ Saved -> {out_path}")


if __name__ == "__main__":
    main()
