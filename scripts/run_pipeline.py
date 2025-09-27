"""CLI entrypoint for running the crypto price insights pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_dashboard.pipeline import run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the crypto price insights pipeline")
    parser.add_argument("--coin-id", default="bitcoin", help="CoinGecko coin identifier")
    parser.add_argument("--vs-currency", default="usd", help="Quote currency")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history")
    parser.add_argument(
        "--export",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "latest_enriched.csv",
        help="CSV export path",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Running pipeline for %s/%s over %s days", parsed.coin_id, parsed.vs_currency, parsed.days)
    results = run_pipeline(
        coin_id=parsed.coin_id,
        vs_currency=parsed.vs_currency,
        days=parsed.days,
        export_path=parsed.export,
    )
    logging.info("Exported enriched dataset to %s", parsed.export)
    logging.info("Key metrics: %s", results["metrics"])


if __name__ == "__main__":
    main()
