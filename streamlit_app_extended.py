"""Extended Streamlit dashboard with ML predictions."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st
from pathlib import Path

from src.crypto_dashboard.pipeline import run_pipeline
from src.crypto_dashboard.ml import DirectionPredictor

st.set_page_config(page_title="Crypto Price Insights", layout="wide")

st.title("📊 Crypto Price Insights Dashboard")
st.sidebar.header("Configuration")

# Main controls
coin_id = st.sidebar.text_input("CoinGecko coin id", value="bitcoin")
vs_currency = st.sidebar.selectbox("Quote currency", options=["usd", "eur", "gbp"], index=0)
days = st.sidebar.slider("Lookback window (days)", min_value=7, max_value=365, value=90)
interval = st.sidebar.selectbox("API interval", options=["", "daily"], index=0)

# ML section
st.sidebar.subheader("🤖 Machine Learning")
enable_ml = st.sidebar.checkbox("Enable ML predictions", value=False)
train_new_model = st.sidebar.button("Train New Model", disabled=not enable_ml)

run_button = st.sidebar.button("Run Analysis", type="primary")

@st.cache_data(show_spinner=True)
def run_cached_pipeline(coin: str, vs_cur: str, period: int, interval_opt: str | None):
    interval_value = interval_opt or None
    results = run_pipeline(coin_id=coin, vs_currency=vs_cur, days=period, interval=interval_value)
    # Convert DataFrames to preserve caching friendliness
    serialisable = {
        "raw": results["raw"].reset_index().to_dict(orient="list"),
        "enriched": results["enriched"].reset_index().to_dict(orient="list"),
        "metrics": results["metrics"],
        "weekend_summary": results["weekend_summary"].reset_index().to_dict(orient="list"),
        "insights": results["insights"],
    }
    return serialisable

@st.cache_data(show_spinner="Training ML model...")
def train_ml_model(enriched_data: pd.DataFrame, coin: str):
    """Train a new ML model."""
    try:
        from src.crypto_dashboard.ml import train_direction_predictor
        
        predictor, results = train_direction_predictor(enriched_data)
        
        return {
            "predictor": predictor,
            "results": results,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "predictor": None,
            "results": None,
            "success": False,
            "error": str(e)
        }

if run_button:
    with st.spinner("Fetching data and computing insights..."):
        cached = run_cached_pipeline(coin_id, vs_currency, days, interval)
        enriched = pd.DataFrame(cached["enriched"])
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"])
        enriched = enriched.set_index("timestamp")
        weekend_summary = pd.DataFrame(cached["weekend_summary"])
        metrics = cached["metrics"]
        insights = cached["insights"]

    # Main metrics display
    st.subheader("📈 Key Metrics")
    metric_cols = st.columns(len(metrics))
    for (name, value), col in zip(metrics.items(), metric_cols):
        col.metric(name.replace("_", " ").title(), f"{value:.4f}")

    # ML Predictions Section
    if enable_ml:
        st.subheader("🤖 Machine Learning Predictions")
        
        ml_col1, ml_col2 = st.columns([2, 1])
        
        with ml_col1:
            if train_new_model:
                with st.spinner("Training machine learning model..."):
                    ml_results = train_ml_model(enriched, coin_id)
                
                if ml_results["success"]:
                    predictor = ml_results["predictor"]
                    results = ml_results["results"]
                    
                    # Make prediction
                    direction, confidence = predictor.predict(enriched)
                    
                    # Display prediction
                    direction_text = "📈 UP" if direction == 1 else "📉 DOWN"
                    st.success(f"**Next Day Prediction**: {direction_text}")
                    st.info(f"**Confidence**: {confidence:.1%}")
                    
                    # Model performance
                    st.write("**Model Performance:**")
                    perf_cols = st.columns(3)
                    perf_cols[0].metric("Test Accuracy", f"{results['metrics']['test_accuracy']:.3f}")
                    perf_cols[1].metric("CV Score", f"{results['metrics']['cv_mean']:.3f}")
                    perf_cols[2].metric("Baseline", f"{results['metrics']['baseline_accuracy']:.3f}")
                    
                else:
                    st.error(f"ML training failed: {ml_results['error']}")
            else:
                st.info("Click 'Train New Model' to generate ML predictions")
        
        with ml_col2:
            st.write("**Model Info:**")
            st.write("- Gradient Boosting Classifier")
            st.write("- Next-day direction prediction") 
            st.write("- Uses technical indicators")
            st.write("- Cross-validated performance")

    # Insights
    st.subheader("💡 Narrative Insights")
    for insight in insights:
        st.markdown(f"- {insight}")

    # Price chart
    st.subheader("📊 Price History")
    chart_data = enriched[["price"]].copy()
    if len(enriched.columns) > 1:
        # Add moving averages if available
        ma_cols = [col for col in enriched.columns if col.startswith("rolling_mean_")]
        for col in ma_cols[:2]:  # Show first 2 moving averages
            chart_data[col.replace("rolling_mean_", "MA")] = enriched[col]
    
    st.line_chart(chart_data)

    # Feature snapshot
    st.subheader("🔧 Feature Snapshot")
    display_cols = ["price", "return", "rolling_mean_7", "rolling_mean_30", "rolling_volatility_7"]
    available_cols = [col for col in display_cols if col in enriched.columns]
    st.dataframe(enriched[available_cols].tail(10).round(4))

    # Weekend vs Weekday analysis
    st.subheader("📅 Weekend vs Weekday Returns")
    if not weekend_summary.empty:
        st.table(weekend_summary.round(4))
    
    # Additional charts
    if len(enriched) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Returns Distribution")
            if "return" in enriched.columns:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                enriched["return"].dropna().hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title("Daily Returns Distribution")
                ax.set_xlabel("Return")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Volatility Trend")
            if "rolling_volatility_7" in enriched.columns:
                st.line_chart(enriched["rolling_volatility_7"].dropna())

else:
    st.info("👆 Adjust the parameters and click **Run Analysis** to fetch the latest data.")
    
    # Show example insights
    st.subheader("🎯 What This Dashboard Provides")
    st.markdown("""
    - **Real-time crypto data** from CoinGecko API
    - **Technical analysis** with moving averages and volatility
    - **Weekend vs weekday** performance patterns
    - **Machine learning predictions** for next-day direction
    - **Automated insights** in plain English
    - **Interactive visualizations** for deep exploration
    """)
    
    # GitHub Actions info
    st.subheader("⚙️ Automation Features")
    st.markdown("""
    This project includes GitHub Actions workflows for:
    - **Daily pipeline runs** to keep insights fresh
    - **ML model training** on-demand
    - **Automated data collection** and storage
    - **Portfolio-ready deployment** capabilities
    """)