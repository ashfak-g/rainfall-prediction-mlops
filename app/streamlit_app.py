"""Streamlit Web Dashboard for Rainfall Prediction System.
Provides Express (3-feature) rapid testing, Detailed (10-feature) forecast,
threshold sensitivity controls, batch CSV scoring, and algorithm benchmarks.
"""

import sys
from pathlib import Path

# Ensure src package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.config import EXPRESS_FEATURES, TOP_10_FEATURES, DEFAULT_DECISION_THRESHOLD
from src.predict import RainfallPredictor

# Streamlit Page Configuration
st.set_page_config(
    page_title="Rainfall Prediction System | Production ML",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .rain-alert-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 16px;
        border-radius: 8px;
        color: #991B1B;
    }
    .rain-alert-low {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        color: #065F46;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    return RainfallPredictor.get_instance()


def render_prediction_card(result: dict):
    pred = result["prediction"]
    prob_pct = result["rain_probability_pct"]
    th = result["decision_threshold"]

    st.markdown("### 📋 Prediction Outcome")
    c1, c2, c3 = st.columns([1.6, 1, 1])

    with c1:
        if pred == 1:
            st.markdown(
                f"""
                <div class="rain-alert-high">
                    <h3>🌧️ Rain Tomorrow Expected</h3>
                    <p>Significant precipitation likelihood within the next 24-hour cycle. Protective measures recommended.</p>
                    <h4>Rain Probability: <b>{prob_pct}%</b> (Threshold: {th})</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rain-alert-low">
                    <h3>☀️ No Rain Tomorrow Expected</h3>
                    <p>Predominantly dry and clear weather expected across the forecasting area.</p>
                    <h4>Rain Probability: <b>{prob_pct}%</b> (Threshold: {th})</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c2:
        st.metric(
            label="Precipitation Probability",
            value=f"{prob_pct}%",
            delta=f"{'+' if prob_pct >= (th * 100) else '-'}{abs(prob_pct - (th * 100)):.1f}% vs Threshold",
        )
        st.progress(prob_pct / 100.0)

    with c3:
        st.metric(label="Risk Classification", value=result["risk_level"])
        st.caption(f"Status: {result['summary']}")


def main():
    predictor = get_predictor()

    # Sidebar Controls
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/rain.png", width=100)
        st.title("Forecasting Control")
        st.markdown(
            """
            **Production Model**: Advanced CatBoost  
            **Test Accuracy**: **86.25%**  
            **ROC-AUC**: **0.8972**  
            **Domain Features**: Enabled
            """
        )
        st.divider()

        st.subheader("⚙️ Decision Sensitivity")
        threshold_mode = st.radio(
            "Classification Preset:",
            options=["Balanced (0.40 - Recommended)", "Default (0.50)", "High Sensitivity (0.35)"],
            index=0,
            help="Threshold tuning adjusts sensitivity: Lower threshold captures more rain days (higher recall)."
        )

        if "Balanced" in threshold_mode:
            current_threshold = 0.40
        elif "High Sensitivity" in threshold_mode:
            current_threshold = 0.35
        else:
            current_threshold = 0.50

        st.caption(f"Current Decision Cutoff: **{current_threshold:.2f}**")
        st.divider()

        st.subheader("Quick Presets")
        scenario = st.selectbox(
            "Test Scenarios:",
            options=["None", "Heavy Rain Incoming", "Clear Sunny Day", "Mild Overcast"],
            index=0
        )

    # Main Header
    st.markdown('<div class="main-header">🌧️ Australian Rainfall Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Production ML Pipeline combining meteorological physics (Pressure Drop, Moisture Delta) with CatBoost gradient boosting.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ Express Forecast (3 Features)",
        "🎯 Detailed Forecast (10 Features)",
        "📁 Batch Prediction (CSV)",
        "📊 Algorithm Benchmarks"
    ])

    # Tab 1: Express Mode (3 Features)
    with tab1:
        st.markdown("### ⚡ Express 3-Feature Rapid Forecast")
        st.info("💡 **Instant Testing**: Enter just the 3 highest-impact indicators (Humidity, Pressure, Rainfall). The system automatically populates secondary features using historical meteorological medians.")

        exp_vals = {"Humidity3pm": 72.0, "Pressure3pm": 1012.0, "Rainfall": 2.0}
        if scenario == "Heavy Rain Incoming":
            exp_vals = {"Humidity3pm": 92.0, "Pressure3pm": 1002.0, "Rainfall": 15.0}
        elif scenario == "Clear Sunny Day":
            exp_vals = {"Humidity3pm": 28.0, "Pressure3pm": 1024.0, "Rainfall": 0.0}
        elif scenario == "Mild Overcast":
            exp_vals = {"Humidity3pm": 60.0, "Pressure3pm": 1014.0, "Rainfall": 1.0}

        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            h3 = st.slider(
                "1. Humidity at 3 PM (%)",
                min_value=0.0, max_value=100.0,
                value=float(exp_vals["Humidity3pm"]),
                step=1.0,
                help="Afternoon humidity has >21% feature importance."
            )
        with c_exp2:
            p3 = st.slider(
                "2. Pressure at 3 PM (hPa)",
                min_value=980.0, max_value=1045.0,
                value=float(exp_vals["Pressure3pm"]),
                step=0.5,
                help="Low barometric pressure marks incoming storm fronts."
            )
        with c_exp3:
            rf = st.slider(
                "3. Today's Rainfall (mm)",
                min_value=0.0, max_value=150.0,
                value=float(exp_vals["Rainfall"]),
                step=0.5,
                help="Precipitation recorded today."
            )

        if st.button("🚀 Run Express Forecast", type="primary", use_container_width=True, key="btn_express"):
            res = predictor.predict_express(h3, p3, rf, threshold=current_threshold)
            render_prediction_card(res)

    # Tab 2: Detailed Mode (10 Features)
    with tab2:
        st.markdown("### 🎯 Detailed Meteorological Forecast (Top 10 Features)")
        st.caption("Inspect and fine-tune primary meteorological variables.")

        det_vals = {}
        if scenario == "Heavy Rain Incoming":
            det_vals = {
                "Humidity3pm": 92.0, "Pressure3pm": 1002.0, "WindGustSpeed": 65.0,
                "Humidity9am": 95.0, "Rainfall": 15.0, "Sunshine": 0.5,
                "Pressure9am": 1005.0, "Temp3pm": 16.0, "MinTemp": 11.0, "MaxTemp": 19.0
            }
        elif scenario == "Clear Sunny Day":
            det_vals = {
                "Humidity3pm": 28.0, "Pressure3pm": 1024.0, "WindGustSpeed": 22.0,
                "Humidity9am": 42.0, "Rainfall": 0.0, "Sunshine": 11.0,
                "Pressure9am": 1026.0, "Temp3pm": 29.0, "MinTemp": 18.0, "MaxTemp": 32.0
            }

        col_l, col_r = st.columns(2)
        detailed_inputs = {}
        half = len(TOP_10_FEATURES) // 2

        with col_l:
            for item in TOP_10_FEATURES[:half]:
                val = det_vals.get(item["name"], item["default"])
                detailed_inputs[item["name"]] = st.slider(
                    label=item["label"],
                    min_value=float(item["min"]),
                    max_value=float(item["max"]),
                    value=float(val),
                    step=float(item["step"]),
                    help=item["description"],
                    key=f"det_{item['name']}"
                )

        with col_r:
            for item in TOP_10_FEATURES[half:]:
                val = det_vals.get(item["name"], item["default"])
                detailed_inputs[item["name"]] = st.slider(
                    label=item["label"],
                    min_value=float(item["min"]),
                    max_value=float(item["max"]),
                    value=float(val),
                    step=float(item["step"]),
                    help=item["description"],
                    key=f"det_{item['name']}"
                )

        if st.button("🚀 Run Detailed Forecast", type="primary", use_container_width=True, key="btn_detailed"):
            res = predictor.predict(detailed_inputs, threshold=current_threshold)
            render_prediction_card(res)

    # Tab 3: Batch CSV Scoring
    with tab3:
        st.markdown("### 📤 Batch CSV Weather Scoring")
        st.markdown("Upload a CSV dataset containing weather observations to generate batch predictions.")

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows. Sample Preview:")
            st.dataframe(batch_df.head(5))

            if st.button("Score Entire Dataset", type="primary"):
                with st.spinner("Executing batch inference..."):
                    scored_df = predictor.predict_batch(batch_df, threshold=current_threshold)
                    st.success("Scoring complete!")
                    st.dataframe(scored_df.head(10))

                    csv_data = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Scored CSV",
                        data=csv_data,
                        file_name="rainfall_predictions_scored.csv",
                        mime="text/csv"
                    )

    # Tab 4: Benchmarks
    with tab4:
        st.markdown("### 🏆 Comprehensive Algorithm Comparison")
        st.markdown("Performance comparison across all tested models on the Australian Weather Dataset:")

        bench_data = [
            {"Model": "Advanced CatBoost (Final Production)", "Accuracy": "86.25%", "Precision": "77.25%", "Recall": "64.22%*", "F1-Score": "0.6742*", "ROC AUC": "0.8972", "Status": "🥇 Champion (Domain Features)"},
            {"Model": "CatBoost (Baseline)", "Accuracy": "85.51%", "Precision": "73.03%", "Recall": "54.49%", "F1-Score": "0.6241", "ROC AUC": "0.8864", "Status": "🥈 Prior Best"},
            {"Model": "XGBoost", "Accuracy": "85.37%", "Precision": "71.11%", "Recall": "56.79%", "F1-Score": "0.6315", "ROC AUC": "0.8862", "Status": "🥉 Alternative"},
            {"Model": "LightGBM", "Accuracy": "85.11%", "Precision": "70.72%", "Recall": "55.53%", "F1-Score": "0.6221", "ROC AUC": "0.8800", "Status": "Alternative"},
            {"Model": "Random Forest", "Accuracy": "84.72%", "Precision": "66.23%", "Recall": "62.80%", "F1-Score": "0.6447", "ROC AUC": "0.8847", "Status": "Ensemble Baseline"},
            {"Model": "Stacking Ensemble", "Accuracy": "84.09%", "Precision": "63.87%", "Recall": "64.25%", "F1-Score": "0.6406", "ROC AUC": "0.8755", "Status": "Hybrid Ensemble"},
            {"Model": "Gradient Boosting", "Accuracy": "82.38%", "Precision": "58.74%", "Recall": "67.66%", "F1-Score": "0.6289", "ROC AUC": "0.8640", "Status": "Baseline"},
            {"Model": "Logistic Regression", "Accuracy": "78.78%", "Precision": "51.29%", "Recall": "76.20%", "F1-Score": "0.6131", "ROC AUC": "0.8610", "Status": "Linear Baseline"},
            {"Model": "Gaussian Naive Bayes", "Accuracy": "77.33%", "Precision": "49.05%", "Recall": "69.69%", "F1-Score": "0.5757", "ROC AUC": "0.8197", "Status": "Probabilistic Baseline"},
        ]
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True)
        st.caption("* Note: Recall and F1-Score for Advanced CatBoost reflect the balanced decision threshold (0.40).")


if __name__ == "__main__":
    main()
