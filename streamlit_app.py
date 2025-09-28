"""Simplified Streamlit app with working ML functionality."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.crypto_dashboard.api import CoinGeckoClient, MarketDataRequest
from src.crypto_dashboard.processing import prepare_price_frame, add_features, summarize_performance
from src.crypto_dashboard.insights import generate_insights


# Page config
st.set_page_config(
    page_title="🚀 Crypto ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Crypto Price Insights Dashboard with ML")
st.markdown("Real-time cryptocurrency analysis with machine learning predictions")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")
coin_id = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["bitcoin", "ethereum", "cardano", "solana"],
    index=0
)
days = st.sidebar.slider("Days of data", 7, 90, 30)

# ML section
st.sidebar.header("🤖 Machine Learning")
enable_ml = st.sidebar.checkbox("Enable ML Predictions", value=False)

# Cache data fetching
@st.cache_data(ttl=300, show_spinner="Fetching crypto data...")
def fetch_and_process_data(coin: str, days_param: int):
    """Fetch and process cryptocurrency data."""
    try:
        client = CoinGeckoClient()
        request = MarketDataRequest(coin_id=coin, days=days_param)
        data = client.market_chart(request)
        prepared = prepare_price_frame(data)
        enriched = add_features(prepared)
        return enriched, None
    except Exception as e:
        return None, str(e)

# Cache ML training
@st.cache_data(ttl=600, show_spinner="Training ML model...")
def train_ml_model(data_hash: str, coin: str):
    """Train ML model with caching."""
    try:
        from src.crypto_dashboard.ml import train_direction_predictor
        # We can't cache the actual data, so we re-fetch it
        client = CoinGeckoClient()
        request = MarketDataRequest(coin_id=coin, days=60)  # More data for ML
        data = client.market_chart(request)
        prepared = prepare_price_frame(data)
        enriched = add_features(prepared)
        
        predictor, results = train_direction_predictor(enriched)
        
        # Make prediction
        direction, confidence = predictor.predict(enriched)
        
        return {
            "success": True,
            "predictor": predictor,
            "results": results,
            "direction": direction,
            "confidence": confidence,
            "data_shape": enriched.shape
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Main app
enriched, error = fetch_and_process_data(coin_id, days)

if error:
    st.error(f"Failed to fetch data: {error}")
    st.stop()

if enriched is None or enriched.empty:
    st.warning("No data available")
    st.stop()

# Display basic info
col1, col2, col3 = st.columns(3)
latest_price = enriched['price'].iloc[-1]
price_change = ((enriched['price'].iloc[-1] / enriched['price'].iloc[-2]) - 1) * 100
col1.metric("Current Price", f"${latest_price:.2f}", f"{price_change:+.2f}%")
col2.metric("Data Points", len(enriched))
col3.metric("Date Range", f"{enriched.index.min().date()} to {enriched.index.max().date()}")

# ML Predictions
if enable_ml:
    st.subheader("🤖 Machine Learning Predictions")
    
    if st.button("🎯 Generate ML Prediction", type="primary"):
        # Create a simple hash for caching
        data_hash = f"{coin_id}_{len(enriched)}_{enriched.index.max()}"
        
        ml_results = train_ml_model(data_hash, coin_id)
        
        if ml_results["success"]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                direction_text = "📈 UP" if ml_results["direction"] == 1 else "📉 DOWN"
                confidence = ml_results["confidence"]
                
                if ml_results["direction"] == 1:
                    st.success(f"**Next Day Prediction: {direction_text}**")
                else:
                    st.error(f"**Next Day Prediction: {direction_text}**")
                    
                st.info(f"**Confidence: {confidence:.1%}**")
                
                # Model performance
                results = ml_results["results"]
                st.write("**Model Performance:**")
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                perf_col1.metric("Test Accuracy", f"{results['metrics']['test_accuracy']:.3f}")
                perf_col2.metric("CV Score", f"{results['metrics']['cv_mean']:.3f}")
                perf_col3.metric("Training Data", f"{ml_results['data_shape'][0]} points")
                
                # Feature importance
                st.write("**Top 5 Most Important Features:**")
                importance_data = results["feature_importance"]
                sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    st.write(f"{i}. **{feature}**: {importance:.3f}")
            
            with col2:
                st.write("**Model Details:**")
                st.write("🎯 **Algorithm**: Gradient Boosting")
                st.write("📊 **Task**: Binary Classification")
                st.write("⏰ **Horizon**: Next Day Direction")
                st.write("📈 **Features**: Technical Indicators")
                st.write("✅ **Validation**: 5-Fold Cross-Validation")
                
        else:
            st.error(f"ML training failed: {ml_results['error']}")
    else:
        st.info("👆 Click the button above to train a model and get predictions!")

# Price Chart
st.subheader("📈 Price Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=enriched.index,
    y=enriched['price'],
    mode='lines',
    name='Price',
    line=dict(color='#1f77b4', width=2)
))
fig.update_layout(
    title=f"{coin_id.title()} Price Over Time",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Performance metrics
metrics = summarize_performance(enriched)
st.subheader("📊 Performance Summary")
metric_cols = st.columns(len(metrics))
for (name, value), col in zip(metrics.items(), metric_cols):
    col.metric(name.replace("_", " ").title(), f"{value:.4f}")

# Insights
insights = list(generate_insights(coin_id, metrics))
st.subheader("💡 Key Insights")
for insight in insights:
    st.markdown(f"• {insight}")

# Technical Analysis
if st.checkbox("Show Technical Analysis"):
    st.subheader("🔧 Technical Indicators")
    
    # Create tabs for different indicators
    tab1, tab2, tab3 = st.tabs(["Moving Averages", "Volatility", "Momentum"])
    
    with tab1:
        if "rolling_mean_7" in enriched.columns and "rolling_mean_30" in enriched.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=enriched.index, y=enriched['price'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=enriched.index, y=enriched['rolling_mean_7'], name='SMA 7', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=enriched.index, y=enriched['rolling_mean_30'], name='SMA 30', line=dict(color='red')))
            fig.update_layout(title="Price with Moving Averages", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if "rolling_volatility_7" in enriched.columns:
            fig = px.line(enriched, y='rolling_volatility_7', title="7-Day Rolling Volatility")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if "return" in enriched.columns:
            fig = px.histogram(enriched['return'].dropna(), nbins=30, title="Daily Returns Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, scikit-learn, and CoinGecko API")