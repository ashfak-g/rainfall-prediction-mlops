# 🌧️ Rainfall Prediction System (MLOps & Web Deployment)

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI Pipeline](https://github.com/ashfak-g/rainfall-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ashfak-g/rainfall-prediction-mlops/actions)
[![Model](https://img.shields.io/badge/Model-Advanced%20CatBoost-yellow?logo=catboost)](https://catboost.ai/)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.25%25-brightgreen)](https://github.com/)
[![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.8972-blue)](https://github.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-grade Machine Learning solution for next-day precipitation forecasting in Australia (`RainTomorrow`). The system combines **meteorological physics (Barometric Pressure Drops, Moisture Accumulation, Diurnal Temperature Range)** with **CatBoost gradient boosting**, achieving **86.25% test accuracy** and **0.8972 ROC-AUC**.

Includes a dual-mode **Streamlit Web Dashboard** (⚡ Express 3-feature test & 🎯 Detailed 10-feature test) and a high-performance **FastAPI REST API**.

---

## 📌 Table of Contents
- [Project Highlights](#-project-highlights)
- [Comprehensive Algorithm Benchmark](#-comprehensive-algorithm-benchmark)
- [Meteorological Domain Feature Engineering](#-meteorological-domain-feature-engineering)
- [Repository Architecture](#-repository-architecture)
- [🎮 Live Run Guide (Step-by-Step for Anyone)](#-live-run-guide-step-by-step-for-anyone)
- [🧭 Input-to-Output Operational Guide & Test Scenarios](#-input-to-output-operational-guide--test-scenarios)
  - [Understanding Key Meteorological Parameters](#understanding-key-meteorological-parameters)
  - [Scenario 1: Imminent Thunderstorm / Heavy Rain (Rain Tomorrow = YES)](#scenario-1-imminent-thunderstorm--heavy-rain-raintomorrow--yes)
  - [Scenario 2: Clear, Bright Sunny Day (Rain Tomorrow = NO)](#scenario-2-clear-bright-sunny-day-raintomorrow--no)
  - [Scenario 3: Borderline Convective Front & Threshold Tuning](#scenario-3-borderline-convective-front--threshold-tuning)
  - [Scenario 4: REST API Request & Response](#scenario-4-rest-api-request--response)
  - [Scenario 5: Batch CSV Prediction Walkthrough](#scenario-5-batch-csv-prediction-walkthrough)
- [🛠️ Advanced Operations](#️-advanced-operations)
  - [Running Automated Unit Tests](#running-automated-unit-tests)
  - [Retraining the Model](#retraining-the-model)
  - [Docker Deployment](#docker-deployment)
  - [GitHub Actions CI/CD](#github-actions-cicd)
- [👨‍💻 Author & Developer](#-author--developer)
- [📄 License](#-license)

---

## ⚡ Project Highlights

- **⚡ Express 3-Feature Forecast**: Ordinary users only need to provide 3 intuitive inputs:
  1. **Humidity at 3 PM (%)** (accounts for >21% of model decision weight)
  2. **Atmospheric Pressure at 3 PM (hPa)** (low pressure signifies convective storm fronts)
  3. **Today's Rainfall (mm)** (precipitation history)
- **🎯 Meteorological Physics Integration**: Computes `Pressure_Delta` ($Pressure_{3pm} - Pressure_{9am}$), `Humidity_Delta`, `Temp_Range`, and cyclical month harmonics.
- **🎚️ Decision Threshold Sensitivity**: Flexible threshold controls (Normal `0.50`, Balanced `0.40`, High Sensitivity `0.35`) allowing weather services to prioritize higher rainfall detection (Recall up to 69%).
- **🛡️ Flawless Data Pipeline**: Clean train-test split before transformation eliminates data leakage; target NaNs dropped rather than falsely imputed.

---

## 🏆 Comprehensive Algorithm Benchmark

All models were evaluated on the test partition of the Australian Weather Dataset:

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Advanced CatBoost (Final Production)** | **86.25%** | **77.25%** | **64.22%*** | **0.6742*** | **0.8972** | 🏆 **Production Champion (Domain Features)** |
| 🥈 | CatBoost (Original Baseline) | 85.51% | 73.03% | 54.49% | 0.6241 | 0.8864 | Prior Best |
| 🥉 | XGBoost | 85.37% | 71.11% | 56.79% | 0.6315 | 0.8862 | Strong Performer |
| 4 | LightGBM | 85.11% | 70.72% | 55.53% | 0.6221 | 0.8800 | Fast Performer |
| 5 | Random Forest | 84.72% | 66.23% | 62.80% | 0.6447 | 0.8847 | Ensemble Baseline |
| 6 | Stacking Ensemble | 84.09% | 63.87% | 64.25% | 0.6406 | 0.8755 | Multi-model Ensemble |
| 7 | Gradient Boosting | 82.38% | 58.74% | 67.66% | 0.6289 | 0.8640 | Tree Baseline |
| 8 | Logistic Regression | 78.78% | 51.29% | 76.20% | 0.6131 | 0.8610 | Linear Baseline |
| 9 | Gaussian Naive Bayes | 77.33% | 49.05% | 69.69% | 0.5757 | 0.8197 | Probabilistic Baseline |

*\* Recall and F1-Score for Advanced CatBoost reflect the balanced decision threshold ($0.40$), drastically reducing missed rain days.*

---

## 🔬 Meteorological Domain Feature Engineering

| Feature | Mathematical Formula | Physical Weather Significance |
| :--- | :--- | :--- |
| `Pressure_Delta` | $Pressure_{3pm} - Pressure_{9am}$ | Rapid afternoon atmospheric pressure drops indicate incoming cyclonic fronts or thunderstorms. |
| `Humidity_Delta` | $Humidity_{3pm} - Humidity_{9am}$ | Positive moisture differential signals strong surface evaporation and cumulus cloud buildup. |
| `Temp_Range` | $MaxTemp - MinTemp$ | Narrow diurnal temperature spans correlate with heavy cloud cover and moisture saturation. |
| `Month_Sin`, `Month_Cos` | $\sin/\cos(2\pi \times Month / 12)$ | Harmonic representation of cyclical annual weather seasons. |

---

## 📂 Repository Architecture

```text
rainfall-prediction-system/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated GitHub Actions CI/CD workflow
├── .gitignore                          # Clean git ignores for caches, venvs, and logs
├── Dockerfile                          # Production multi-port container setup
├── LICENSE                             # MIT Open Source License
├── README.md                           # Professional documentation & benchmark tables
├── requirements.txt                    # Pinned production dependencies
├── app/                                # User interfaces & web endpoints
│   ├── __init__.py
│   ├── api.py                          # FastAPI REST service with latency tracking & /predict/express
│   └── streamlit_app.py                # Streamlit UI with Express 3-feature mode & sensitivity controls
├── data/
│   ├── processed/
│   │   └── clean_rainfall.csv          # Preprocessed dataset
│   └── raw/
│       └── weatherAUS.csv              # Raw Australian weather observations
├── models/
│   ├── best_catboost_model.pkl         # Production CatBoost model (86.25% accuracy, 0.8972 ROC-AUC)
│   └── scaler.pkl                      # Fitted StandardScaler (30 features, warning-free)
├── notebooks/
│   └── rainfall_prediction_analysis.ipynb # Research & exploratory analysis notebook
├── src/                                # Core modular Python package
│   ├── __init__.py
│   ├── config.py                       # Paths, 30 features, express features, and medians
│   ├── predict.py                      # Production inference engine with threshold tuning
│   ├── preprocessing.py                # Domain feature engineering & express vector builder
│   └── train.py                        # Automated retraining CLI script
└── tests/                              # Automated test suite
    ├── __init__.py
    └── test_prediction.py              # 8 comprehensive Pytest unit & integration tests
```

---

## 🎮 Live Run Guide (Step-by-Step for Anyone)

Anyone can run this project locally in **under 2 minutes** by following these simple steps:

### Step 1: Clone the Repository
```bash
git clone https://github.com/ashfak-g/rainfall-prediction-mlops.git
cd rainfall-prediction-mlops
```

### Step 2: Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Application

You have two interactive ways to run the project:

#### Option A: Interactive Streamlit Web UI (Recommended)
```bash
streamlit run app/streamlit_app.py
```
> 🌐 The application will automatically open in your default browser at **`http://localhost:8501`**.

#### Option B: High-Performance FastAPI REST Server
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
> 📖 Visit the interactive Swagger UI documentation at **`http://localhost:8000/docs`**.

---

## 🧭 Input-to-Output Operational Guide & Test Scenarios

Here is a clear guide explaining how changing meteorological parameters directly determines the machine learning model's prediction outcome.

### Understanding Key Meteorological Parameters

| Parameter | Normal Range | Physical Impact on Rainfall |
| :--- | :--- | :--- |
| **Humidity at 3 PM** (`Humidity3pm`) | `10% - 100%` | **Most influential feature (>21%)**. Afternoon humidity above 70% indicates heavy atmospheric moisture, making rain highly probable. Humidity below 35% almost always yields dry weather. |
| **Pressure at 3 PM** (`Pressure3pm`) | `990 - 1035 hPa` | **Second most critical feature**. Atmospheric pressure below 1010 hPa indicates a low-pressure cyclone/trough. Higher pressure (>1020 hPa) brings stable, clear skies. |
| **Today's Rainfall** (`Rainfall`) | `0.0 - 100+ mm` | Rainfall recorded today. Ongoing rain systems strongly increase the prior probability of rain continuing tomorrow. |
| **Wind Gust Speed** (`WindGustSpeed`) | `15 - 120 km/h` | High wind gusts (>50 km/h) indicate strong convective thunderstorm activity or cold front passage. |
| **Cloud at 3 PM** (`Cloud3pm`) | `0 - 8 oktas` | `8 oktas` represents complete cloud cover (overcast). `0 oktas` means crystal clear sky. |
| **Sunshine Duration** (`Sunshine`) | `0 - 14 hours` | Low sunshine (0-3 hrs) strongly correlates with cloudy/rainy conditions. |

---

### Scenario 1: Imminent Thunderstorm / Heavy Rain (Rain Tomorrow = YES)

When an incoming low-pressure storm front brings heavy moisture and cloudy skies:

#### 📥 Input Settings:
* **Mode**: ⚡ Express Forecast (or 🎯 Detailed Forecast)
* **Humidity at 3 PM**: `85.0%` (Saturated afternoon air)
* **Atmospheric Pressure at 3 PM**: `1004.0 hPa` (Deep barometric depression)
* **Today's Rainfall**: `14.5 mm` (Active precipitation cycle)
* *Optional Detailed Settings*: Cloud 3pm = `8 oktas`, Wind Gust = `55 km/h`, Sunshine = `1.5 hours`

#### 📤 Expected Output:
```text
┌──────────────────────────────────────────────────────────────┐
│  🌧️ Rain Tomorrow Expected                                   │
│  Significant precipitation likelihood within next 24 hours.  │
├──────────────────────────────────────────────────────────────┤
│  Prediction:          YES (Rain Tomorrow)                    │
│  Rain Probability:    88.75%                                 │
│  Risk Level:          🔴 SEVERE / HIGH RISK                  │
│  Decision Threshold:  0.40                                   │
│  Advisory:            Carry an umbrella, expect showers.     │
└──────────────────────────────────────────────────────────────┘
```

---

### Scenario 2: Clear, Bright Sunny Day (Rain Tomorrow = NO)

When a high-pressure anticyclone dominates, bringing warm dry air and cloudless skies:

#### 📥 Input Settings:
* **Mode**: ⚡ Express Forecast (or 🎯 Detailed Forecast)
* **Humidity at 3 PM**: `24.0%` (Very dry air)
* **Atmospheric Pressure at 3 PM**: `1024.5 hPa` (High atmospheric pressure)
* **Today's Rainfall**: `0.0 mm` (Completely dry)
* *Optional Detailed Settings*: Cloud 3pm = `1 okta`, Wind Gust = `22 km/h`, Sunshine = `11.0 hours`

#### 📤 Expected Output:
```text
┌──────────────────────────────────────────────────────────────┐
│  ☀️ No Rain Tomorrow Expected                                │
│  Predominantly dry and clear weather across the area.        │
├──────────────────────────────────────────────────────────────┤
│  Prediction:          NO (Dry Weather)                       │
│  Rain Probability:    5.20%                                  │
│  Risk Level:          🟢 VERY LOW RISK                       │
│  Decision Threshold:  0.40                                   │
│  Advisory:            Optimal conditions for outdoor events. │
└──────────────────────────────────────────────────────────────┘
```

---

### Scenario 3: Borderline Convective Front & Threshold Tuning

In marginal weather conditions, changing the decision threshold slider allows users to tune between false alarms and missed rain:

#### 📥 Input Settings:
* **Humidity at 3 PM**: `58.0%`
* **Atmospheric Pressure at 3 PM**: `1013.0 hPa`
* **Today's Rainfall**: `1.0 mm`
* **Model Calculated Probability**: `44.50%`

#### 📤 Output Comparison by Threshold:
| Decision Preset | Threshold | Output Label | Explanation |
| :--- | :---: | :---: | :--- |
| **Normal / Standard** | `0.50` | ☀️ **No Rain** (`44.5% < 50%`) | Standard conservative prediction. Minimizes false rain warnings. |
| **Balanced (Recommended)** | `0.40` | 🌧️ **Rain Tomorrow** (`44.5% >= 40%`) | **Catches 64.2% of all actual rain days**, preventing surprise downpours for events and agriculture. |
| **High Sensitivity** | `0.35` | 🌧️ **Rain Tomorrow** (`44.5% >= 35%`) | Maximum safety alert mode; triggers warnings even for moderate shower risks. |

---

### Scenario 4: REST API Request & Response

Send an instant HTTP POST request to `/predict/express`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/express' \
  -H 'Content-Type: application/json' \
  -d '{
    "Humidity3pm": 82.0,
    "Pressure3pm": 1006.5,
    "Rainfall": 8.0,
    "threshold": 0.40
  }'
```

#### 📤 JSON Output Returned:
```json
{
  "prediction": 1,
  "label": "Rain Tomorrow",
  "rain_probability": 0.8145,
  "rain_probability_pct": 81.45,
  "decision_threshold": 0.4,
  "risk_level": "Severe / Rain Highly Likely",
  "risk_badge": "SEVERE",
  "summary": "[SEVERE] Rain Tomorrow (81.45% probability at threshold 0.4)",
  "inference_latency_ms": 2.41
}
```

---

### Scenario 5: Batch CSV Prediction Walkthrough

1. In the Streamlit sidebar, select **📁 Batch CSV Scoring**.
2. Upload any CSV containing weather observations (e.g., `data/raw/weatherAUS.csv` or a custom test file).
3. Click **🚀 Generate Batch Predictions**.
4. The system automatically appends:
   - `Predicted_RainTomorrow` (0 or 1)
   - `Prediction_Label` ("Rain Tomorrow" or "No Rain Tomorrow")
   - `Rain_Probability_Pct` (e.g., 78.4%)
5. Click **⬇️ Download Enriched CSV** to save results.

---

## 🛠️ Advanced Operations

### Running Automated Unit Tests
Verify model persistence, singleton stability, edge-case handling, and API endpoints:
```bash
pytest -v tests/test_prediction.py
```
*Expected result: `8 passed in ~1.5s`*

### Retraining the Model
Retrain the model from raw data with custom hyperparameters:
```bash
python src/train.py --iterations 600 --lr 0.06 --depth 6
```

### Docker Deployment
Run both UI and API in an isolated container:
```bash
# Build Docker image
docker build -t rainfall-prediction-mlops:latest .

# Run container (Maps Streamlit to port 8501)
docker run -d -p 8501:8501 --name rainfall-app rainfall-prediction-mlops:latest
```

### GitHub Actions CI/CD
Whenever code is pushed to `main`, GitHub Actions automatically:
1. Provisions virtual environments across **Python 3.10, 3.11, and 3.12**.
2. Installs dependencies and verifies compatibility.
3. Executes the full **pytest** suite to prevent any regression.

---

## 👨‍💻 Author & Developer

<div align="center">

### **Ashfakur Rahman**
🎓 **B.Sc. in Computer Science and Engineering**  
🏛️ **Green University of Bangladesh**

[![GitHub Profile](https://img.shields.io/badge/GitHub-ashfak--g-181717?style=for-the-badge&logo=github)](https://github.com/ashfak-g)
[![Email](https://img.shields.io/badge/Email-ashfakgub221%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ashfakgub221@gmail.com)

</div>

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
