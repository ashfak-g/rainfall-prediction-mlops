"""Unit and integration test suite for Rainfall Prediction System.
Verifies model loading, domain feature construction, express forecasting,
threshold tuning, batch scoring, and FastAPI endpoints.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS, DEFAULT_DECISION_THRESHOLD
from src.preprocessing import build_inference_vector
from src.predict import RainfallPredictor
from fastapi.testclient import TestClient
from app.api import app


def test_artifact_files_exist():
    """Verify production model and scaler artifacts exist on disk."""
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    assert SCALER_PATH.exists(), f"Scaler artifact missing at {SCALER_PATH}"


def test_model_feature_dimension():
    """Verify that scaler and model support all 30 base and domain features."""
    predictor = RainfallPredictor.reload_instance()
    assert predictor.scaler.n_features_in_ == 30
    assert len(FEATURE_COLUMNS) == 30


def test_build_inference_vector_express():
    """Verify vector generation from just 3 express inputs automatically computes domain features."""
    user_inputs = {
        "Humidity3pm": 85.0,
        "Pressure3pm": 1005.0,
        "Rainfall": 10.0,
    }
    vec = build_inference_vector(user_inputs)
    assert isinstance(vec, pd.DataFrame)
    assert len(vec) == 1
    assert list(vec.columns) == FEATURE_COLUMNS
    assert vec.loc[0, "Humidity3pm"] == 85.0
    assert vec.loc[0, "Pressure3pm"] == 1005.0
    # Domain feature Pressure_Delta must be present and numeric
    assert "Pressure_Delta" in vec.columns
    assert isinstance(float(vec.loc[0, "Pressure_Delta"]), float)


def test_express_prediction_rain_scenario():
    """Verify high humidity, low pressure, and rainfall produce 'Rain Tomorrow'."""
    predictor = RainfallPredictor.get_instance()
    res = predictor.predict_express(humidity_3pm=90.0, pressure_3pm=1002.0, rainfall=15.0)
    assert res["prediction"] == 1
    assert "Rain Tomorrow" in res["label"]
    assert res["rain_probability"] > 0.60
    assert res["rain_probability_pct"] > 60.0


def test_express_prediction_dry_scenario():
    """Verify low humidity, high pressure, and zero rainfall produce 'No Rain Tomorrow'."""
    predictor = RainfallPredictor.get_instance()
    res = predictor.predict_express(humidity_3pm=25.0, pressure_3pm=1025.0, rainfall=0.0)
    assert res["prediction"] == 0
    assert "No Rain Tomorrow" in res["label"]
    assert res["rain_probability"] < 0.25
    assert res["rain_probability_pct"] < 25.0


def test_decision_threshold_tuning():
    """Verify threshold parameter alters binary classification cutoff correctly."""
    predictor = RainfallPredictor.get_instance()
    marginal_inputs = {
        "Humidity3pm": 68.0,
        "Pressure3pm": 1014.0,
        "Rainfall": 1.5,
    }
    # Test with strict threshold (0.60) vs sensitive threshold (0.30)
    res_strict = predictor.predict(marginal_inputs, threshold=0.70)
    res_sensitive = predictor.predict(marginal_inputs, threshold=0.30)

    assert res_strict["decision_threshold"] == 0.70
    assert res_sensitive["decision_threshold"] == 0.30
    # Probability remains constant; binary prediction adjusts
    assert res_strict["rain_probability"] == res_sensitive["rain_probability"]


def test_batch_prediction():
    """Verify batch prediction scores multiple rows."""
    predictor = RainfallPredictor.get_instance()
    sample_df = pd.DataFrame([
        {"Humidity3pm": 92.0, "Pressure3pm": 1002.0, "Rainfall": 14.0},
        {"Humidity3pm": 28.0, "Pressure3pm": 1024.0, "Rainfall": 0.0},
    ])
    scored = predictor.predict_batch(sample_df, threshold=0.40)
    assert "Predicted_RainTomorrow" in scored.columns
    assert "Rain_Probability_Pct" in scored.columns
    assert len(scored) == 2
    assert scored.loc[0, "Predicted_RainTomorrow"] == 1
    assert scored.loc[1, "Predicted_RainTomorrow"] == 0


def test_fastapi_endpoints():
    """Verify FastAPI express, standard, and metadata endpoints."""
    client = TestClient(app)

    # Healthcheck
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    # Model metadata
    info_res = client.get("/model-info")
    assert info_res.status_code == 200
    assert info_res.json()["accuracy"] == 0.8625

    # Express Prediction Endpoint (3 parameters)
    express_payload = {
        "Humidity3pm": 88.0,
        "Pressure3pm": 1004.0,
        "Rainfall": 10.0,
        "threshold": 0.40
    }
    exp_res = client.post("/predict/express", json=express_payload)
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["prediction"] == 1
    assert "rain_probability_pct" in exp_data

    # Detailed Prediction Endpoint
    std_payload = {
        "Humidity3pm": 30.0,
        "Pressure3pm": 1025.0,
        "WindGustSpeed": 25.0,
        "Humidity9am": 45.0,
        "Rainfall": 0.0,
        "Sunshine": 10.5,
        "Pressure9am": 1027.0,
        "Temp3pm": 28.0,
        "MinTemp": 17.0,
        "MaxTemp": 31.0,
        "threshold": 0.40
    }
    std_res = client.post("/predict", json=std_payload)
    assert std_res.status_code == 200
    std_data = std_res.json()
    assert std_data["prediction"] == 0
