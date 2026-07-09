"""End-to-end ML trading-signal pipeline (offline: data -> backtest -> charts).

    python run_pipeline.py --ticker AAPL --model random_forest
    python run_pipeline.py --ticker QQQ --years 5 --threshold 0.6

Steps: download 5y daily OHLCV -> engineer features -> PCA(>=80% var) ->
train an ML classifier -> generate the Long/Flat signal -> backtest the test
window against Buy & Hold -> print metrics -> save charts + metrics.csv.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless: save charts without a display

import pandas as pd

from src.backtest import run_backtest
from src.data import get_daily_ohlcv
from src.metrics import metrics_table
from src.model import train_signal_model
from src.plotting import (
    plot_drawdowns, plot_equity_curves, plot_pca_variance, plot_signal,
)

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")


def main() -> None:
    ap = argparse.ArgumentParser(description="ML trading-signal pipeline (Alpaca data)")
    ap.add_argument("--ticker", default="F", help="Ticker (AAPL, MSFT, SPY, QQQ, NVDA, …)")
    ap.add_argument("--years", type=int, default=5, help="Years of history (>=5)")
    ap.add_argument("--model", default="gradient_boosting",
                    help="random_forest | logistic | gradient_boosting | svm | mlp")
    ap.add_argument("--threshold", type=float, default=0.60, help="Long if P(up) > threshold")
    ap.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print(f"[1/5] Downloading {args.years}y of daily data for {ticker}…")
    df = get_daily_ohlcv(ticker, years=args.years)
    print(f"      {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")

    print(f"[2/5] Engineering features, fitting PCA, training {args.model}…")
    sm = train_signal_model(df, model=args.model, threshold=args.threshold)
    print(f"      PCA kept {sm.pca.n_components} components "
          f"({sm.pca.explained_variance_ratio[:sm.pca.n_components].sum() * 100:.1f}% variance)")
    print(f"      Test window: {sm.test_index[0].date()} → {sm.test_index[-1].date()} "
          f"({len(sm.test_index)} days)")

    long_days = int(sm.signal.sum())
    print(f"      Signal is LONG on {long_days}/{len(sm.signal)} test days")

    print("[3/5] Backtesting ML signal vs Buy & Hold on the test window…")
    df_test = sm.test_frame()
    ml_result = run_backtest(df_test, sm.signal, initial_capital=args.capital)
    bh_signal = pd.Series(1, index=df_test.index)  # always invested
    bh_result = run_backtest(df_test, bh_signal, initial_capital=args.capital)
    results = {"ML Signal": ml_result, "Buy & Hold": bh_result}

    print("[4/5] Performance comparison:\n")
    table = metrics_table(results)
    print(table.to_string())
    table.to_csv(os.path.join(CHARTS_DIR, f"{ticker}_metrics.csv"))

    print("\n[5/5] Saving charts…")
    plot_pca_variance(sm.pca, save=os.path.join(CHARTS_DIR, f"{ticker}_pca_variance.png"))
    plot_equity_curves(results, save=os.path.join(CHARTS_DIR, f"{ticker}_equity.png"))
    plot_drawdowns(results, save=os.path.join(CHARTS_DIR, f"{ticker}_drawdown.png"))
    plot_signal(df_test, sm.proba, sm.signal, sm.threshold,
                f"{ticker} — {args.model} Long/Flat signal",
                save=os.path.join(CHARTS_DIR, f"{ticker}_signal.png"))
    print(f"      Charts + metrics saved to {CHARTS_DIR}/")
    print("\nDone. (Data is from Alpaca; PAPER trading only — no real money.)")


if __name__ == "__main__":
    main()
