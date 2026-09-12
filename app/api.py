"""FastAPI Production REST API for Rainfall Prediction System.
Provides OpenAPI / Swagger documented endpoints for express, detailed, and batch predictions
with latency tracking and structured logging.
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure src is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import TOP_10_FEATURES, FEATURE_COLUMNS, DEFAULT_DECISION_THRESHOLD
from src.predict import RainfallPredictor

# Setup production logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RainfallAPI")

app = FastAPI(
    title="Rainfall Prediction REST API",
    description="Production-grade Machine Learning API forecasting next-day Australian rainfall using Advanced CatBoost.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Tracks request processing duration and logs endpoint access."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Method={request.method} Path={request.url.path} Status={response.status_code} Latency={duration_ms:.2f}ms")
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# Pydantic Request & Response Schemas
class ExpressWeatherInput(BaseModel):
    Humidity3pm: float = Field(72.0, ge=0.0, le=100.0, description="Relative humidity at 3 PM (%)")
    Pressure3pm: float = Field(1012.0, ge=950.0, le=1060.0, description="Atmospheric pressure at 3 PM (hPa)")
    Rainfall: float = Field(2.0, ge=0.0, le=400.0, description="Precipitation today (mm)")
    threshold: Optional[float] = Field(default=DEFAULT_DECISION_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold (default: 0.40)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Humidity3pm": 88.0,
                "Pressure3pm": 1004.0,
                "Rainfall": 12.0,
                "threshold": 0.40
            }
        }
    }


class WeatherInput(BaseModel):
    Humidity3pm: float = Field(70.0, ge=0.0, le=100.0, description="Relative humidity at 3 PM (%)")
    Pressure3pm: float = Field(1012.0, ge=950.0, le=1060.0, description="Atmospheric pressure at 3 PM (hPa)")
    WindGustSpeed: float = Field(40.0, ge=0.0, le=200.0, description="Speed of strongest wind gust (km/h)")
    Humidity9am: float = Field(65.0, ge=0.0, le=100.0, description="Relative humidity at 9 AM (%)")
    Rainfall: float = Field(2.0, ge=0.0, le=400.0, description="Precipitation today (mm)")
    Sunshine: float = Field(6.0, ge=0.0, le=24.0, description="Hours of bright sunshine")
    Pressure9am: float = Field(1015.0, ge=950.0, le=1060.0, description="Atmospheric pressure at 9 AM (hPa)")
    Temp3pm: float = Field(22.0, ge=-20.0, le=60.0, description="Temperature at 3 PM (°C)")
    MinTemp: float = Field(12.0, ge=-20.0, le=50.0, description="Minimum temperature (°C)")
    MaxTemp: float = Field(25.0, ge=-10.0, le=60.0, description="Maximum temperature (°C)")
    threshold: Optional[float] = Field(default=DEFAULT_DECISION_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold")
    extra_features: Optional[Dict[str, float]] = Field(default=None, description="Optional extra features")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Humidity3pm": 85.0,
                "Pressure3pm": 1005.0,
                "WindGustSpeed": 55.0,
                "Humidity9am": 90.0,
                "Rainfall": 8.5,
                "Sunshine": 1.2,
                "Pressure9am": 1008.0,
                "Temp3pm": 18.0,
                "MinTemp": 13.0,
                "MaxTemp": 20.0,
                "threshold": 0.40
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Rain, 1 = Rain Tomorrow")
    label: str = Field(..., description="Human readable outcome")
    rain_probability: float = Field(..., description="Probability from 0.0 to 1.0")
    rain_probability_pct: float = Field(..., description="Probability as percentage")
    decision_threshold: float = Field(..., description="Cutoff threshold used")
    risk_level: str = Field(..., description="Risk tier")
    risk_badge: str = Field(..., description="Risk category code")
    summary: str = Field(..., description="Formatted summary string")


class BatchInput(BaseModel):
    records: List[WeatherInput]
    threshold: Optional[float] = Field(default=DEFAULT_DECISION_THRESHOLD, ge=0.0, le=1.0)


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    dataset: str
    accuracy: float
    precision: float
    roc_auc: float
    default_threshold: float
    total_features_supported: int
    express_features: List[str]


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Rainfall Prediction API",
        "status": "online",
        "version": "2.0.0",
        "documentation": "/docs"
    }


@app.get("/health", tags=["General"])
def health():
    try:
        predictor = RainfallPredictor.get_instance()
        return {
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "feature_dimension": predictor.scaler.n_features_in_
        }
    except Exception as e:
        logger.error(f"Healthcheck failed: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_info():
    return ModelInfoResponse(
        model_name="Advanced CatBoost Rainfall Forecaster",
        model_type="CatBoostClassifier",
        dataset="Australian Daily Weather Observations (weatherAUS)",
        accuracy=0.8625,
        precision=0.7725,
        roc_auc=0.8972,
        default_threshold=DEFAULT_DECISION_THRESHOLD,
        total_features_supported=len(FEATURE_COLUMNS),
        express_features=["Humidity3pm", "Pressure3pm", "Rainfall"]
    )


@app.post("/predict/express", response_model=PredictionResponse, tags=["Inference"])
def predict_express_endpoint(input_data: ExpressWeatherInput):
    """Rapid forecast using only 3 highest-priority indicators (Humidity3pm, Pressure3pm, Rainfall)."""
    try:
        predictor = RainfallPredictor.get_instance()
        th = input_data.threshold if input_data.threshold is not None else DEFAULT_DECISION_THRESHOLD
        result = predictor.predict_express(
            humidity_3pm=input_data.Humidity3pm,
            pressure_3pm=input_data.Pressure3pm,
            rainfall=input_data.Rainfall,
            threshold=th
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Express prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_standard(input_data: WeatherInput):
    """Detailed forecast using standard meteorological inputs."""
    try:
        predictor = RainfallPredictor.get_instance()
        features = input_data.model_dump(exclude_unset=True)
        th = features.pop("threshold", DEFAULT_DECISION_THRESHOLD)
        if "extra_features" in features and features["extra_features"]:
            extra = features.pop("extra_features")
            features.update(extra)

        result = predictor.predict(features, threshold=th)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Standard prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch_api(batch_data: BatchInput):
    """Batch prediction on multiple weather records."""
    try:
        predictor = RainfallPredictor.get_instance()
        results = []
        th = batch_data.threshold if batch_data.threshold is not None else DEFAULT_DECISION_THRESHOLD
        for record in batch_data.records:
            features = record.model_dump(exclude_unset=True)
            rec_th = features.pop("threshold", th)
            if "extra_features" in features and features["extra_features"]:
                extra = features.pop("extra_features")
                features.update(extra)
            res = predictor.predict(features, threshold=rec_th)
            results.append(PredictionResponse(**res))
        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
