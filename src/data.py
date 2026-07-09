"""Historical OHLCV download from Alpaca's Market Data API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .config import Settings, load_settings


def get_daily_ohlcv(symbol: str, years: int = 5, settings: Settings | None = None) -> pd.DataFrame:
    """Download at least ``years`` years of daily OHLCV bars for ``symbol``.

    Returns a DataFrame indexed by date with columns:
    open, high, low, close, volume.
    """
    settings = settings or load_settings()
    client = StockHistoricalDataClient(settings.api_key, settings.secret_key)
    feed = DataFeed.SIP if settings.data_feed.lower() == "sip" else DataFeed.IEX

    start = datetime.now(timezone.utc) - timedelta(days=int(years * 365.25) + 5)
    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame.Day,
        start=start,
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol!r}.")

    df = df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol.upper()].drop(columns="symbol")
    df = df.rename(columns={"timestamp": "date"}).set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]
