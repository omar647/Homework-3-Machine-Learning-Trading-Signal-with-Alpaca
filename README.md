# 🤖 Machine-Learning Trading Signal with Alpaca (Paper Trading Only)

**Homework 3** — Build a simple ML trading signal on Alpaca market data,
evaluate it with a backtest, and demonstrate it in Alpaca's **paper** trading
environment.

> ⚠️ **PAPER TRADING ONLY — no real money is used.**

## What it does

- Pulls **5 years of daily OHLCV** for any ticker from Alpaca.
- Engineers **11 technical indicators** (trend / momentum / volatility / volume)
  plus **log returns** and **rolling mean & std** — 29 features total.
- Standardizes features and runs **PCA**, keeping the components that explain
  **≥ 80%** of the variance.
- Trains an **ML classifier** (Gradient Boosting by default) on the PCA components.
  Target = *next-day return > 0*. Signal = **Long** if P(up) > 0.60, else **Flat**.
- **Backtests** the signal on a held-out test window ($100k, long-only, no
  leverage, no shorting) and compares it to **Buy & Hold**.
- Reports **Total Return, CAGR, Volatility, Sharpe, Sortino, Max Drawdown,
  Win Rate**.
- Submits a real **paper trade** to Alpaca and logs the signal + order.

## Project structure

```
homework3-ml-signal/
├── run_pipeline.py          # CLI: data → features → PCA → ML → backtest → charts
├── paper_trade.py           # live Alpaca PAPER trading demo (submits an order)
├── ml_report.ipynb          # report notebook (charts + metrics inline)
├── requirements.txt
├── .env.example             # copy to .env, add Alpaca PAPER keys
├── .gitignore
├── charts/                  # generated PNGs + metrics.csv (committed deliverable)
└── src/
    ├── config.py            # load API keys from environment
    ├── data.py              # Alpaca daily OHLCV download
    ├── indicators.py        # 11 indicators (SMA, EMA, MACD, ADX, RSI, …)
    ├── features.py          # feature matrix: indicators + log returns + rolling stats
    ├── pca.py               # StandardScaler + PCA (≥80% variance)
    ├── model.py             # ML model, target, Long/Flat signal
    ├── backtest.py          # long-only backtesting engine
    ├── metrics.py           # performance metrics + comparison table
    └── plotting.py          # PCA variance, equity, drawdown, signal charts
```

## Setup

```bash
pip3 install -r requirements.txt
cp .env.example .env          # add your Alpaca PAPER keys (they start with "PK")
```

> On macOS use `python3` / `pip3` (not `python`).

## Run

**Offline pipeline** (downloads data, prints metrics, saves charts to `charts/`):

```bash
python3 run_pipeline.py                       # defaults: SPY, gradient_boosting
python3 run_pipeline.py --ticker AAPL         # choose any ticker
python3 run_pipeline.py --ticker QQQ --threshold 0.6
```

The user chooses the ticker with `--ticker` (AAPL, MSFT, SPY, QQQ, NVDA, …).
Default model is **Gradient Boosting**; also available: `random_forest`,
`logistic`, `svm`, `mlp` (via `--model`).

**Report notebook** (charts inline):

```bash
jupyter notebook ml_report.ipynb
```

**Paper trading demo** (submits a PAPER order to Alpaca):

```bash
python3 paper_trade.py --qty 5                          # follow the live signal
python3 paper_trade.py --ticker NVDA --qty 5            # choose any ticker
python3 paper_trade.py --qty 5 --force buy              # guarantee a demo trade
```

## Features

| Category | Features |
|---|---|
| Trend | SMA(20,50), EMA(20,50,200), MACD (+signal, +hist), ADX (+DI, −DI) |
| Momentum | RSI, Stochastic %K/%D, Williams %R |
| Volatility | Bollinger Bands (+width), ATR |
| Volume | OBV, CMF |
| Returns | Log returns, rolling mean & std (5, 10, 20) |

## PCA

Features are standardized (`StandardScaler`), then `PCA` keeps the smallest
number of components reaching **≥ 80%** cumulative variance — for SPY that is
**4 components (~83%)**. PCA is fit on the **training window only** so the test
backtest has no look-ahead leakage. See `charts/*_pca_variance.png`.

## Example results (SPY, Gradient Boosting, test window Jan 2025 – Jul 2026)

```bash
python3 run_pipeline.py                       # defaults: SPY, gradient_boosting
```

| | Total Return | CAGR | Volatility | Sharpe | Sortino | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|---|---|---|
| **ML Signal** | −1.56% | −1.09% | 9.22% | −0.07 | −0.04 | **−10.42%** | 63.64% | 44 |
| **Buy & Hold** | +23.00% | 15.46% | 17.24% | 0.92 | 1.20 | −18.98% | 100.00% | 1 |

Over this window SPY trended steadily higher, so Buy & Hold wins on total
return — a realistic outcome for a long/flat signal on an index in a bull run.
The signal does keep **about half the volatility and a much smaller drawdown**
(−10.4% vs −19.0%) by sitting in cash on weak days. Different tickers, models,
and thresholds give very different results — try a few:

```bash
python3 run_pipeline.py --ticker AAPL --model mlp
python3 run_pipeline.py --ticker QQQ  --model gradient_boosting
```

*(Past performance is not indicative of future results; this is an educational
exercise.)*

## Charts (`charts/`)

- `*_pca_variance.png` — PCA scree + cumulative variance with the 80% cutoff
- `*_signal.png` — price with Long/Flat shading + the P(up) probability track
- `*_equity.png` — equity curve: ML Signal vs Buy & Hold
- `*_drawdown.png` — drawdown comparison
- `*_metrics.csv` — the metrics table

## Paper trading demo — proof

`paper_trade.py` connects to the Alpaca **paper** endpoint
(`TradingClient(..., paper=True)`), scores the latest bar, and submits an order.
Example log:

```
[HH:MM:SS] Connected to PAPER account PA3GL5RGH9D9 (status=ACTIVE, cash=$100,000.00)
[HH:MM:SS] Latest bar 2026-07-08  close=$745.28  P(up)=0.104  → signal=FLAT
[HH:MM:SS] Decision: BUY
[HH:MM:SS] ORDER SUBMITTED → BUY 10.0 SPY (id=38602b55-…, status=ACCEPTED)
```

*(Run with `--force buy` for a guaranteed demo order when the latest signal is
FLAT, or plain `python3 paper_trade.py` to act on the live signal.)*

*(Add your Alpaca paper dashboard screenshot showing the executed order here.)*

## Video

[![Watch the demo](https://img.youtube.com/vi/-NZ54breupE/hqdefault.jpg)](https://youtu.be/-NZ54breupE)
[![Watch the demo](https://img.youtube.com/vi/CVu4DqwVwTI/hqdefault.jpg)](https://youtu.be/CVu4DqwVwTI)


**"This is paper trading only — no real money is used."**

---

> ⚠️ **Reminder: this homework uses Alpaca PAPER TRADING only. No real money.**
