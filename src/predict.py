"""Inference engine for Rainfall Prediction System.
Loads trained CatBoost model and StandardScaler to make real-time express, standard, and batch predictions.
"""

import sys
from typing import Dict, Any, Union, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS,
    DEFAULT_DECISION_THRESHOLD,
)
from src.preprocessing import build_inference_vector


class RainfallPredictor:
    """Production predictor for next-day Australian rainfall with threshold sensitivity control."""

    _instance: Optional["RainfallPredictor"] = None

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH, scaler_path: Union[str, Path] = SCALER_PATH):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self._load_artifacts()

    @classmethod
    def get_instance(cls, model_path: Union[str, Path] = MODEL_PATH, scaler_path: Union[str, Path] = SCALER_PATH) -> "RainfallPredictor":
        """Singleton accessor to avoid redundant model deserialization across requests."""
        if cls._instance is None:
            cls._instance = cls(model_path, scaler_path)
        return cls._instance

    @classmethod
    def reload_instance(cls) -> "RainfallPredictor":
        """Forces reload of model artifacts from disk (useful after retraining)."""
        cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Loads serialized model and scaler from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at: {self.scaler_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def predict(
        self,
        input_features: Dict[str, Any],
        threshold: float = DEFAULT_DECISION_THRESHOLD
    ) -> Dict[str, Any]:
        """Performs rainfall prediction for weather inputs with threshold sensitivity.
        
        Args:
            input_features: Dictionary containing weather observations (e.g. 3 express, 10 standard, or full 30).
            threshold: Probability decision threshold (0.0 to 1.0). Default 0.40 for balanced high recall.
            
        Returns:
            Dictionary containing prediction outcome, probability, threshold, and risk tier.
        """
        # Construct feature vector with intelligent defaults
        df_vector = build_inference_vector(input_features)

        # Scale features
        scaled_features = self.scaler.transform(df_vector)

        # Calculate precipitation probability
        if hasattr(self.model, "predict_proba"):
            proba_arr = self.model.predict_proba(scaled_features)[0]
            rain_proba = float(proba_arr[1])
        else:
            raw_pred = int(self.model.predict(scaled_features)[0])
            rain_proba = 1.0 if raw_pred == 1 else 0.0

        # Apply decision threshold
        binary_pred = 1 if rain_proba >= threshold else 0
        rain_pct = round(rain_proba * 100.0, 2)

        # Risk Tier Classification
        if rain_proba < 0.25:
            risk_level = "Very Low Risk"
            badge = "LOW"
            emoji_icon = "🟢"
        elif rain_proba < 0.45:
            risk_level = "Moderate Risk"
            badge = "MODERATE"
            emoji_icon = "🟡"
        elif rain_proba < 0.70:
            risk_level = "High Risk"
            badge = "HIGH"
            emoji_icon = "🟠"
        else:
            risk_level = "Severe / Rain Highly Likely"
            badge = "SEVERE"
            emoji_icon = "🔴"

        label = "Rain Tomorrow" if binary_pred == 1 else "No Rain Tomorrow"

        return {
            "prediction": binary_pred,
            "label": label,
            "rain_probability": round(rain_proba, 4),
            "rain_probability_pct": rain_pct,
            "decision_threshold": threshold,
            "risk_level": risk_level,
            "risk_badge": badge,
            "summary": f"[{badge}] {label} ({rain_pct}% probability at threshold {threshold})"
        }

    # Express prediction interface to be added in next iteration

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = DEFAULT_DECISION_THRESHOLD
    ) -> pd.DataFrame:
        """Performs batch predictions on a DataFrame.
        
        Args:
            df: DataFrame containing meteorological features.
            threshold: Probability threshold for positive classification.
            
        Returns:
            DataFrame with added 'Predicted_RainTomorrow' and 'Rain_Probability' columns.
        """
        from src.preprocessing import compute_domain_features, DEFAULT_FEATURE_MEDIANS

        clean_df = df.copy()
        clean_df = compute_domain_features(clean_df)

        for col in FEATURE_COLUMNS:
            if col not in clean_df.columns:
                clean_df[col] = DEFAULT_FEATURE_MEDIANS.get(col, 0.0)

        clean_df = clean_df[FEATURE_COLUMNS]
        scaled = self.scaler.transform(clean_df)

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(scaled)[:, 1]
        else:
            probas = self.model.predict(scaled)

        preds = (probas >= threshold).astype(int)

        result_df = df.copy()
        result_df["Predicted_RainTomorrow"] = preds
        result_df["Rain_Probability_Pct"] = np.round(probas * 100, 2)
        result_df["Prediction_Label"] = np.where(preds == 1, "Rain Tomorrow", "No Rain Tomorrow")
        return result_df
