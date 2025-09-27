"""Feature engineering and descriptive statistics helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def prepare_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate the price frame and ensure expected schema."""
    if "price" not in prices.columns:
        raise ValueError("Price dataframe must contain a 'price' column")

    frame = prices.copy()
    frame = frame.sort_index()
    frame.index.name = "timestamp"
    frame = frame[~frame.index.duplicated(keep="first")]
    frame["price"] = frame["price"].astype(float)
    return frame


def add_features(frame: pd.DataFrame, rolling_windows: Iterable[int] = (7, 30)) -> pd.DataFrame:
    """Add derived metrics such as returns, rolling averages, and volatility."""

    enriched = frame.copy()
    enriched["return"] = enriched["price"].pct_change()
    enriched["log_return"] = np.log(enriched["price"]).diff()

    for window in rolling_windows:
        enriched[f"rolling_mean_{window}"] = enriched["price"].rolling(window=window, min_periods=1).mean()
        enriched[f"rolling_std_{window}"] = enriched["price"].rolling(window=window, min_periods=2).std()
        enriched[f"rolling_volatility_{window}"] = enriched["return"].rolling(window=window, min_periods=2).std() * np.sqrt(365)

    enriched["cum_return"] = (1 + enriched["return"].fillna(0)).cumprod() - 1
    enriched["drawdown"] = enriched["price"].div(enriched["price"].cummax()) - 1
    return enriched


def summarize_performance(frame: pd.DataFrame) -> dict[str, float]:
    """Return descriptive statistics for portfolio storytelling."""

    cleaned = frame.dropna(subset=["price"]).copy()
    if cleaned.empty:
        raise ValueError("Cannot summarise an empty dataframe")

    latest_price = cleaned["price"].iloc[-1]
    avg_price = cleaned["price"].mean()
    median_price = cleaned["price"].median()

    returns = cleaned["return"].dropna()
    daily_vol = returns.std() if not returns.empty else 0.0
    annual_vol = daily_vol * np.sqrt(365)
    cumulative_return = cleaned["cum_return"].dropna().iloc[-1] if "cum_return" in cleaned else 0.0
    max_drawdown = cleaned["drawdown"].min() if "drawdown" in cleaned else 0.0

    return {
        "latest_price": float(latest_price),
        "average_price": float(avg_price),
        "median_price": float(median_price),
        "daily_volatility": float(daily_vol),
        "annualised_volatility": float(annual_vol),
        "cumulative_return": float(cumulative_return),
        "max_drawdown": float(max_drawdown),
    }


def weekend_weekday_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise average returns by weekday vs. weekend."""

    enriched = frame.copy()
    enriched["weekday"] = enriched.index.day_name()
    enriched["is_weekend"] = enriched.index.weekday >= 5
    summary = (
        enriched.groupby("is_weekend")["return"].agg(["mean", "std", "count"]).rename(index={True: "weekend", False: "weekday"})
    )
    return summary
