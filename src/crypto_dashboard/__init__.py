"""Crypto Price Insights Dashboard package."""

from .api import CoinGeckoClient
from .processing import (
    add_features,
    prepare_price_frame,
    summarize_performance,
    weekend_weekday_split,
)
from .insights import generate_insights
from .pipeline import run_pipeline

__all__ = [
    "CoinGeckoClient",
    "add_features",
    "prepare_price_frame",
    "summarize_performance",
    "weekend_weekday_split",
    "generate_insights",
    "run_pipeline",
]
