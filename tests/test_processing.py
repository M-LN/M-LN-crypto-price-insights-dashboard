"""Unit tests for feature engineering helpers."""

from __future__ import annotations

import pandas as pd

from src.crypto_dashboard.processing import add_features, prepare_price_frame, summarize_performance


def _sample_prices() -> pd.DataFrame:
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "price": [40000, 40500, 41000, 42000, 41500, 43000, 43500, 44000, 44500, 45000],
    }
    frame = pd.DataFrame(data).set_index("timestamp")
    return frame


def test_prepare_price_frame_returns_sorted_index():
    frame = _sample_prices().sample(frac=1, random_state=42)
    prepared = prepare_price_frame(frame)
    assert prepared.index.is_monotonic_increasing
    assert prepared.iloc[0]["price"] == 40000


def test_add_features_creates_expected_columns():
    frame = prepare_price_frame(_sample_prices())
    enriched = add_features(frame, rolling_windows=(3,))
    expected_cols = {"return", "log_return", "rolling_mean_3", "rolling_std_3", "rolling_volatility_3", "cum_return", "drawdown"}
    assert expected_cols.issubset(enriched.columns)


def test_summarize_performance_outputs_metrics():
    frame = prepare_price_frame(_sample_prices())
    enriched = add_features(frame)
    metrics = summarize_performance(enriched.dropna())
    assert {"latest_price", "cumulative_return", "annualised_volatility"}.issubset(metrics.keys())
