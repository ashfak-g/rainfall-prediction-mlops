"""Configuration module for Rainfall Prediction System.
Defines paths, feature lists, domain transformations, statistical defaults, and metadata.
"""

from pathlib import Path
import numpy as np

# Project Directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "weatherAUS.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "clean_rainfall.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_catboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

TARGET_COL = "RainTomorrow"

# Optimal classification threshold for balanced precision and high recall
DEFAULT_DECISION_THRESHOLD = 0.40

# Base meteorological features from raw weather observations
BASE_FEATURE_COLUMNS = [
    "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "Year", "Month", "Day"
]

# Engineered meteorological domain features
DOMAIN_FEATURE_COLUMNS = [
    "Pressure_Delta",  # Pressure3pm - Pressure9am (sudden pressure drops indicate incoming storms)
    "Humidity_Delta",  # Humidity3pm - Humidity9am (afternoon moisture accumulation)
    "Temp_Range",      # MaxTemp - MinTemp (diurnal temperature range)
    "Temp_Delta",      # Temp3pm - Temp9am (daytime warming rate)
    "Month_Sin",       # Cyclical sin representation of month
    "Month_Cos"        # Cyclical cos representation of month
]

# Full feature matrix used by the production model (Base + Domain)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DOMAIN_FEATURE_COLUMNS

# Baseline medians calculated across the entire dataset (clean_rainfall.csv)
# Used to impute missing parameters during online inference when user supplies a subset
DEFAULT_FEATURE_MEDIANS = {
    "Location": 24.0,
    "MinTemp": 12.0,
    "MaxTemp": 22.6,
    "Rainfall": 0.0,
    "Evaporation": 4.8,
    "Sunshine": 8.4,
    "WindGustDir": 9.0,
    "WindGustSpeed": 39.0,
    "WindDir9am": 7.0,
    "WindDir3pm": 8.0,
    "WindSpeed9am": 13.0,
    "WindSpeed3pm": 19.0,
    "Humidity9am": 70.0,
    "Humidity3pm": 52.0,
    "Pressure9am": 1017.6,
    "Pressure3pm": 1015.2,
    "Cloud9am": 5.0,
    "Cloud3pm": 5.0,
    "Temp9am": 16.7,
    "Temp3pm": 21.1,
    "RainToday": 0.0,
    "Year": 2013.0,
    "Month": 6.0,
    "Day": 16.0,
    "Pressure_Delta": -2.4,
    "Humidity_Delta": -18.0,
    "Temp_Range": 10.6,
    "Temp_Delta": 4.4,
    "Month_Sin": 0.0,
    "Month_Cos": -1.0
}

# Express Mode: The top 3 most decisive features for rapid user testing
EXPRESS_FEATURES = [
    {
        "name": "Humidity3pm",
        "label": "Humidity at 3 PM (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 72.0,
        "step": 1.0,
        "description": "Relative humidity in the afternoon (accounts for >21% of model decision weight)"
    },
    {
        "name": "Pressure3pm",
        "label": "Atmospheric Pressure at 3 PM (hPa)",
        "min": 980.0,
        "max": 1045.0,
        "default": 1012.0,
        "step": 0.5,
        "description": "Barometric pressure at 3 PM (low pressure signifies convective storm fronts)"
    },
    {
        "name": "Rainfall",
        "label": "Today's Rainfall (mm)",
        "min": 0.0,
        "max": 150.0,
        "default": 2.0,
        "step": 0.5,
        "description": "Precipitation recorded today (rain today increases prior probability of continuing rain)"
    }
]

# Standard Mode: Top 10 features for detailed meteorological evaluation
TOP_10_FEATURES = [
    {
        "name": "Humidity3pm",
        "label": "Humidity at 3 PM (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 70.0,
        "step": 1.0,
        "description": "Relative humidity measured at 3:00 PM"
    },
    {
        "name": "Pressure3pm",
        "label": "Atmospheric Pressure at 3 PM (hPa)",
        "min": 980.0,
        "max": 1045.0,
        "default": 1012.0,
        "step": 0.5,
        "description": "Atmospheric pressure reduced to mean sea level at 3:00 PM"
    },
    {
        "name": "WindGustSpeed",
        "label": "Wind Gust Speed (km/h)",
        "min": 0.0,
        "max": 140.0,
        "default": 40.0,
        "step": 1.0,
        "description": "Speed of strongest wind gust in the 24 hours to midnight"
    },
    {
        "name": "Humidity9am",
        "label": "Humidity at 9 AM (%)",
        "min": 0.0,
        "max": 100.0,
        "default": 65.0,
        "step": 1.0,
        "description": "Relative humidity measured at 9:00 AM"
    },
    {
        "name": "Rainfall",
        "label": "Today's Rainfall (mm)",
        "min": 0.0,
        "max": 300.0,
        "default": 2.0,
        "step": 0.5,
        "description": "Amount of rain recorded for the day in millimeters"
    },
    {
        "name": "Sunshine",
        "label": "Sunshine Duration (hours)",
        "min": 0.0,
        "max": 15.0,
        "default": 6.0,
        "step": 0.5,
        "description": "Number of hours of bright sunshine in the day"
    },
    {
        "name": "Pressure9am",
        "label": "Atmospheric Pressure at 9 AM (hPa)",
        "min": 980.0,
        "max": 1045.0,
        "default": 1015.0,
        "step": 0.5,
        "description": "Atmospheric pressure at 9:00 AM"
    },
    {
        "name": "Temp3pm",
        "label": "Temperature at 3 PM (°C)",
        "min": -10.0,
        "max": 50.0,
        "default": 22.0,
        "step": 0.5,
        "description": "Temperature measured at 3:00 PM"
    },
    {
        "name": "MinTemp",
        "label": "Minimum Temperature (°C)",
        "min": -10.0,
        "max": 35.0,
        "default": 12.0,
        "step": 0.5,
        "description": "The minimum temperature in degrees Celsius"
    },
    {
        "name": "MaxTemp",
        "label": "Maximum Temperature (°C)",
        "min": 0.0,
        "max": 50.0,
        "default": 25.0,
        "step": 0.5,
        "description": "The maximum temperature in degrees Celsius"
    }
]
