"""Streamlit dashboard prototype for the crypto price insights project."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.crypto_dashboard.pipeline import run_pipeline

st.set_page_config(page_title="Crypto Price Insights", layout="wide")

st.title("📊 Crypto Price Insights Dashboard")

with st.sidebar:
    st.header("Configuration")
    coin_id = st.text_input("CoinGecko coin id", value="bitcoin")
    vs_currency = st.selectbox("Quote currency", options=["usd", "eur", "gbp"], index=0)
    days = st.slider("Lookback window (days)", min_value=7, max_value=365, value=90)
    interval = st.selectbox("API interval", options=["", "daily"], index=0)
    run_button = st.button("Run analysis")

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

if run_button:
    with st.spinner("Fetching data and computing insights..."):
        cached = run_cached_pipeline(coin_id, vs_currency, days, interval)
        enriched = pd.DataFrame(cached["enriched"])
        enriched["timestamp"] = pd.to_datetime(enriched["timestamp"])
        enriched = enriched.set_index("timestamp")
        weekend_summary = pd.DataFrame(cached["weekend_summary"])
        metrics = cached["metrics"]
        insights = cached["insights"]

    st.subheader("Key Metrics")
    metric_cols = st.columns(len(metrics))
    for (name, value), col in zip(metrics.items(), metric_cols):
        col.metric(name.replace("_", " ").title(), f"{value:.4f}")

    st.subheader("Narrative Insights")
    for insight in insights:
        st.markdown(f"- {insight}")

    st.subheader("Price History")
    st.line_chart(enriched["price"])

    st.subheader("Feature Snapshot")
    st.dataframe(enriched.tail(10))

    st.subheader("Weekend vs Weekday Returns")
    if not weekend_summary.empty:
        st.table(weekend_summary)
else:
    st.info("Adjust the parameters and click *Run analysis* to fetch the latest data.")
