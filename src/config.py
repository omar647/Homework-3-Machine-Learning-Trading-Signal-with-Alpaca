"""Load Alpaca API keys from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    secret_key: str
    data_feed: str = "iex"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)


def load_settings() -> Settings:
    s = Settings(
        api_key=os.getenv("ALPACA_API_KEY", "").strip(),
        secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip(),
        data_feed=os.getenv("ALPACA_DATA_FEED", "iex").strip() or "iex",
    )
    if not s.is_configured:
        raise RuntimeError(
            "Alpaca API keys not found. Copy .env.example to .env and set "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY (paper keys)."
        )
    return s
