"""Chart helpers for the crypto dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def price_trend_chart(frame: pd.DataFrame, coin: str):
    """Return a Matplotlib figure showing price and rolling average."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.index, frame["price"], label=f"{coin.upper()} price", color="#1f77b4")
    rolling_cols = [col for col in frame.columns if col.startswith("rolling_mean_")]
    for col in rolling_cols:
        ax.plot(frame.index, frame[col], label=col.replace("_", " "), linestyle="--")
    ax.set_title(f"{coin.title()} price history")
    ax.set_ylabel("Price (quote currency)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def interactive_price_chart(frame: pd.DataFrame, coin: str):
    """Return a Plotly figure for interactive exploration."""

    fig = px.line(frame.reset_index(), x="timestamp", y="price", title=f"{coin.title()} price history")
    fig.update_layout(hovermode="x unified")
    return fig


def returns_histogram(frame: pd.DataFrame, coin: str):
    """Return a Plotly histogram of daily returns."""

    fig = px.histogram(
        frame.reset_index(),
        x="return",
        nbins=40,
        title=f"{coin.title()} daily returns distribution",
        labels={"return": "Daily return"},
    )
    fig.update_layout(bargap=0.05)
    return fig
