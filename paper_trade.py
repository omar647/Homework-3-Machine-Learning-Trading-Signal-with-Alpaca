"""Live paper-trading demo — Alpaca PAPER environment ONLY. No real money.

Pipeline for one run:
    fetch latest daily data -> engineer features -> PCA -> ML signal
    -> submit a PAPER order:
         signal = LONG  -> BUY  (if not already holding)
         signal = FLAT  -> SELL (if currently holding)

Usage:
    python paper_trade.py --ticker AAPL --qty 5
    python paper_trade.py --ticker AAPL --qty 5 --force buy   # guarantee a demo trade

Safety: this uses Alpaca's paper endpoint (paper-api.alpaca.markets) via
``TradingClient(..., paper=True)``. It can never touch a live account.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.config import load_settings
from src.data import get_daily_ohlcv
from src.model import train_signal_model


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def current_qty(client: TradingClient, symbol: str) -> float:
    try:
        pos = client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0  # no open position


def submit(client: TradingClient, symbol: str, qty: float, side: OrderSide):
    req = MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY
    )
    order = client.submit_order(req)
    log(f"ORDER SUBMITTED → {side.value.upper()} {qty} {symbol} "
        f"(id={order.id}, status={order.status})")
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpaca PAPER trading demo for the ML signal")
    ap.add_argument("--ticker", default="F", help="Ticker to trade")
    ap.add_argument("--qty", type=float, default=5, help="Share quantity per order")
    ap.add_argument("--years", type=int, default=5, help="Years of history for the model")
    ap.add_argument("--model", default="gradient_boosting", help="ML model to use")
    ap.add_argument("--threshold", type=float, default=0.60, help="Long if P(up) > threshold")
    ap.add_argument("--force", choices=["auto", "buy", "sell"], default="auto",
                    help="auto = follow the model; buy/sell = force an order for the demo")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    settings = load_settings()

    print("=" * 64)
    print("  ALPACA PAPER TRADING DEMO — NO REAL MONEY IS USED")
    print("=" * 64)

    # Trading client explicitly pinned to the PAPER endpoint.
    client = TradingClient(settings.api_key, settings.secret_key, paper=True)
    acct = client.get_account()
    log(f"Connected to PAPER account {acct.account_number} "
        f"(status={acct.status}, cash=${float(acct.cash):,.2f})")

    log(f"Fetching latest {args.years}y daily data for {ticker}…")
    df = get_daily_ohlcv(ticker, years=args.years)

    log(f"Training {args.model} + PCA and scoring the latest bar…")
    sm = train_signal_model(df, model=args.model, threshold=args.threshold)
    sig = sm.latest_signal()
    log(f"Latest bar {sig['date'].date()}  close=${sig['close']:.2f}  "
        f"P(up)={sig['probability']:.3f}  → signal={sig['label']}")

    held = current_qty(client, ticker)
    log(f"Current {ticker} position: {held} shares")

    # Decide the action.
    if args.force == "buy":
        action = "BUY"
    elif args.force == "sell":
        action = "SELL"
    elif sig["signal"] == 1 and held == 0:
        action = "BUY"
    elif sig["signal"] == 0 and held > 0:
        action = "SELL"
    else:
        action = "HOLD"

    log(f"Decision: {action}")

    if action == "BUY":
        submit(client, ticker, args.qty, OrderSide.BUY)
    elif action == "SELL":
        qty = held if held > 0 else args.qty
        submit(client, ticker, qty, OrderSide.SELL)
    else:
        log("No order submitted (signal already matches the current position).")

    print("=" * 64)
    print("  Reminder: PAPER TRADING ONLY — this is not real money.")
    print("=" * 64)


if __name__ == "__main__":
    main()
