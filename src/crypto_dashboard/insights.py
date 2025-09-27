"""Narrative insight helpers for portfolio storytelling."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def generate_insights(coin: str, metrics: dict[str, float], weekend_split: pd.DataFrame | None = None) -> Iterable[str]:
    """Create a list of narrative-friendly insight strings."""

    if "annualised_volatility" in metrics:
        yield (
            f"{coin.title()} experienced an annualised volatility of "
            f"{metrics['annualised_volatility']:.2%} over the selected period."
        )

    if {"cumulative_return", "latest_price"}.issubset(metrics):
        yield (
            f"Total return for {coin.title()} reached {metrics['cumulative_return']:.2%}, "
            f"with the latest price at ${metrics['latest_price']:.2f}."
        )

    if "max_drawdown" in metrics:
        yield (
            f"The maximum drawdown observed for {coin.title()} was {metrics['max_drawdown']:.2%}, "
            "framing the downside risk profile."
        )

    if weekend_split is not None and not weekend_split.empty:
        weekend_mean = weekend_split.loc["weekend", "mean"] if "weekend" in weekend_split.index else None
        weekday_mean = weekend_split.loc["weekday", "mean"] if "weekday" in weekend_split.index else None
        if weekend_mean is not None and weekday_mean is not None:
            if weekend_mean > weekday_mean:
                intro = "Weekend sessions outperformed weekdays"
            else:
                intro = "Weekday sessions outperformed weekends"
            yield (
                f"{intro}: average weekend return {weekend_mean:.2%} vs weekday {weekday_mean:.2%}."
            )
