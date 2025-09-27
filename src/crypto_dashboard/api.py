"""API client helpers for fetching crypto market data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"


@dataclass
class MarketDataRequest:
    """Parameters accepted by the CoinGecko market chart endpoint."""

    coin_id: str
    vs_currency: str = "usd"
    days: int | str = 30
    interval: Optional[str] = None  # "daily" can reduce payload for long ranges


class CoinGeckoClient:
    """Minimal CoinGecko market data client."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def market_chart(self, request: MarketDataRequest) -> pd.DataFrame:
        """Fetch historical market data and return a tidy price dataframe.

        Parameters
        ----------
        request:
            MarketDataRequest containing the API parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by timestamp with a single ``price`` column.
        """

        url = f"{self.base_url}/coins/{request.coin_id}/market_chart"
        params = {
            "vs_currency": request.vs_currency,
            "days": request.days,
        }
        if request.interval:
            params["interval"] = request.interval

        logger.info("Requesting CoinGecko market chart", extra={"params": params})
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        prices = payload.get("prices")
        if not prices:
            raise ValueError("No price data returned from CoinGecko")

        frame = pd.DataFrame(prices, columns=["timestamp", "price"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()

        # Normalise timezone to naive UTC for easier downstream consumption
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        frame["price"] = frame["price"].astype(float)
        return frame
