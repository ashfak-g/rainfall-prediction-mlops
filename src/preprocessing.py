"""Data Preprocessing Pipeline for Rainfall Prediction System.
Handles data cleaning, missing value imputation, domain feature engineering, and inference vector generation.
"""

import sys
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    BASE_FEATURE_COLUMNS,
    DOMAIN_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    DEFAULT_FEATURE_MEDIANS,
    TARGET_COL,
)


def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes meteorological domain features from basic weather metrics.
    
    Features engineered:
    - Pressure_Delta: Afternoon vs morning pressure differential
    - Humidity_Delta: Afternoon vs morning humidity differential
    - Temp_Range: Daily maximum vs minimum temperature span
    - Temp_Delta: Afternoon vs morning temperature change
    - Month_Sin & Month_Cos: Cyclical calendar features
    
    Args:
        df: DataFrame containing base columns (Pressure, Humidity, Temp, Month).
        
    Returns:
        DataFrame with added domain feature columns.
    """
    df_out = df.copy()

    # Pressure change
    if "Pressure3pm" in df_out.columns and "Pressure9am" in df_out.columns:
        df_out["Pressure_Delta"] = df_out["Pressure3pm"] - df_out["Pressure9am"]
    else:
        df_out["Pressure_Delta"] = DEFAULT_FEATURE_MEDIANS["Pressure_Delta"]

    # Humidity change
    if "Humidity3pm" in df_out.columns and "Humidity9am" in df_out.columns:
        df_out["Humidity_Delta"] = df_out["Humidity3pm"] - df_out["Humidity9am"]
    else:
        df_out["Humidity_Delta"] = DEFAULT_FEATURE_MEDIANS["Humidity_Delta"]

    # Diurnal temperature range
    if "MaxTemp" in df_out.columns and "MinTemp" in df_out.columns:
        df_out["Temp_Range"] = df_out["MaxTemp"] - df_out["MinTemp"]
    else:
        df_out["Temp_Range"] = DEFAULT_FEATURE_MEDIANS["Temp_Range"]

    # Daytime temperature change
    if "Temp3pm" in df_out.columns and "Temp9am" in df_out.columns:
        df_out["Temp_Delta"] = df_out["Temp3pm"] - df_out["Temp9am"]
    else:
        df_out["Temp_Delta"] = DEFAULT_FEATURE_MEDIANS["Temp_Delta"]

    # Cyclical seasonality
    if "Month" in df_out.columns:
        df_out["Month_Sin"] = np.sin(2 * np.pi * df_out["Month"] / 12)
        df_out["Month_Cos"] = np.cos(2 * np.pi * df_out["Month"] / 12)
    else:
        df_out["Month_Sin"] = DEFAULT_FEATURE_MEDIANS["Month_Sin"]
        df_out["Month_Cos"] = DEFAULT_FEATURE_MEDIANS["Month_Cos"]

    return df_out


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleans raw Australian Weather dataset.
    
    Methodology:
    - Drops rows where target RainTomorrow is missing (eliminates label noise)
    - Maps binary target (Yes -> 1, No -> 0)
    - Parses Date to Year, Month, Day
    - Computes domain meteorological features
    - Label encodes categorical features
    - Imputes missing numerical values with median and categoricals with mode
    
    Args:
        df: Raw pandas DataFrame read from weatherAUS.csv
        
    Returns:
        X: Feature matrix with 30 features (Base + Domain)
        y: Target series (RainTomorrow, 0 or 1)
    """
    df_clean = df.copy()

    # Drop rows missing the target variable
    if TARGET_COL in df_clean.columns:
        df_clean = df_clean.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df_clean[TARGET_COL] = df_clean[TARGET_COL].map({"No": 0, "Yes": 1})
        y = df_clean[TARGET_COL]
    else:
        y = pd.Series()

    if "RainToday" in df_clean.columns:
        df_clean["RainToday"] = df_clean["RainToday"].map({"No": 0, "Yes": 1}).fillna(0)

    # Date parsing
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])
        df_clean["Year"] = df_clean["Date"].dt.year
        df_clean["Month"] = df_clean["Date"].dt.month
        df_clean["Day"] = df_clean["Date"].dt.day
        df_clean = df_clean.drop(columns=["Date"])

    # Compute domain features
    df_clean = compute_domain_features(df_clean)

    # Impute numerical features
    for col in df_clean.select_dtypes(include=np.number).columns:
        if col != TARGET_COL:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Impute and encode categorical features
    for col in df_clean.select_dtypes(include=["object", "category", "string"]).columns:
        mode_val = df_clean[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
        df_clean[col] = df_clean[col].fillna(fill_val)
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Drop target from feature matrix
    drop_cols = [TARGET_COL] if TARGET_COL in df_clean.columns else []
    X = df_clean.drop(columns=drop_cols)

    # Ensure exact column ordering
    available_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    X = X[available_cols]

    return X, y


def build_inference_vector(
    user_inputs: Dict[str, Any],
    medians: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Builds a single-sample feature DataFrame aligned with FEATURE_COLUMNS.
    
    Supports Express Mode (e.g. only 3 features provided) as well as full feature dictionaries.
    Unspecified features are automatically populated using statistical baselines.
    
    Args:
        user_inputs: Dictionary containing meteorological inputs.
        medians: Optional baseline medians mapping.
        
    Returns:
        1-row DataFrame containing all features required by the production model.
    """
    baseline = (medians or DEFAULT_FEATURE_MEDIANS).copy()

    # Overwrite baseline with user-specified inputs
    for key, val in user_inputs.items():
        if key in baseline and val is not None:
            baseline[key] = float(val)

    # Dynamically recompute domain features based on provided or baseline inputs
    df_temp = pd.DataFrame([baseline])
    df_temp = compute_domain_features(df_temp)

    # Return exactly the required columns in exact order
    return df_temp[FEATURE_COLUMNS]
