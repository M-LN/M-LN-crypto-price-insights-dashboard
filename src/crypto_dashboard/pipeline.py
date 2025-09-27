"""Pipeline orchestration for the crypto price insights project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .api import CoinGeckoClient, MarketDataRequest
from .insights import generate_insights
from .processing import add_features, prepare_price_frame, summarize_performance, weekend_weekday_split


def run_pipeline(
    coin_id: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 30,
    interval: Optional[str] = None,
    export_path: Optional[Path] = None,
) -> dict[str, object]:
    """Fetch, enrich, and summarise crypto prices."""

    client = CoinGeckoClient()
    request = MarketDataRequest(coin_id=coin_id, vs_currency=vs_currency, days=days, interval=interval)
    raw_prices = client.market_chart(request)
    prepared = prepare_price_frame(raw_prices)
    enriched = add_features(prepared)
    metrics = summarize_performance(enriched.dropna(subset=["return"]))
    weekend_summary = weekend_weekday_split(enriched.dropna(subset=["return"]))
    insights = list(generate_insights(coin_id, metrics, weekend_summary))

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(export_path, index=True)

    return {
        "raw": raw_prices,
        "prepared": prepared,
        "enriched": enriched,
        "metrics": metrics,
        "weekend_summary": weekend_summary,
        "insights": insights,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the crypto price insights pipeline")
    parser.add_argument("--coin-id", default="bitcoin", help="CoinGecko coin identifier (e.g. bitcoin, ethereum)")
    parser.add_argument("--vs-currency", default="usd", help="Quote currency (usd, eur, etc.)")
    parser.add_argument("--days", type=int, default=30, help="Number of lookback days")
    parser.add_argument("--interval", default=None, help="Optional API interval (e.g. daily)")
    parser.add_argument("--export", type=Path, default=None, help="Optional CSV export path")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(args)
    results = run_pipeline(
        coin_id=parsed.coin_id,
        vs_currency=parsed.vs_currency,
        days=parsed.days,
        interval=parsed.interval,
        export_path=parsed.export,
    )

    print("Pipeline complete! Key metrics:")
    for key, value in results["metrics"].items():
        print(f" - {key}: {value:.4f}")

    print("\nGenerated insights:")
    for insight in results["insights"]:
        print(f" • {insight}")


if __name__ == "__main__":
    main()
