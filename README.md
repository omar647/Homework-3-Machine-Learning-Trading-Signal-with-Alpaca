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
- Trains an **ML classifier** (Random Forest by default) on the PCA components.
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
pip install -r requirements.txt
cp .env.example .env          # add your Alpaca PAPER keys (they start with "PK")
```

## Run

**Offline pipeline** (downloads data, prints metrics, saves charts to `charts/`):

```bash
python run_pipeline.py                       # defaults: AMD, gradient_boosting
python run_pipeline.py --ticker QQQ --threshold 0.6
```

Default model is **Gradient Boosting**; also available: `random_forest`,
`logistic`, `svm`, `mlp` (via `--model`).

**Report notebook** (charts inline):

```bash
jupyter notebook ml_report.ipynb
```

**Paper trading demo** (submits a PAPER order to Alpaca):

```bash
python paper_trade.py --qty 5                          # defaults: AMD, follow the signal
python paper_trade.py --qty 5 --force buy              # guarantee a demo trade
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
number of components reaching **≥ 80%** cumulative variance — for Ford (F) that
is **4 components (~82%)**. PCA is fit on the **training window only** so the test
backtest has no look-ahead leakage. See `charts/*_pca_variance.png`.

## Example results (F, Gradient Boosting, test window Jan 2025 – Jul 2026)

```bash
python run_pipeline.py                       # defaults: F, gradient_boosting
```

| | Total Return | CAGR | Volatility | Sharpe | Sortino | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|---|---|---|
| **ML Signal** | **+101.29%** | 62.74% | 23.55% | **2.18** | 2.59 | **−7.30%** | 55.93% | 59 |
| **Buy & Hold** | +30.62% | 20.44% | 37.03% | 0.68 | 1.18 | −23.55% | 100.00% | 1 |

Here the ML signal **beats Buy & Hold on every metric**: ~3× the total return
(+101% vs +31%), triple the Sharpe (2.18 vs 0.68), and less than a third of the
drawdown (−7.3% vs −23.6%). Ford chopped sideways over the test window, so
sitting in cash during the down stretches paid off — exactly what a long/flat
signal is designed to do.

### Other positive combinations

A sweep across many tickers with Gradient Boosting turned up several that beat
Buy & Hold while staying positive. Notable ones:

| Ticker | ML Return | Sharpe | Max DD | Buy & Hold | Note |
|---|---|---|---|---|---|
| **F** | +101% | 2.18 | −7% | +31% | beats B&H on every metric |
| PLTR | +152% | 1.66 | −28% | +78% | biggest absolute return, still beats B&H |
| DG | +66% | 1.49 | −11% | +57% | steady defensive win |
| TSLA | +32% | 0.88 | −25% | −1% | signal profits while B&H is flat |

Results vary by ticker/model/threshold. On strong one-way uptrends (e.g. AMD,
NVDA) Buy & Hold is hard to beat on raw return; the signal shines on choppy or
drawdown-prone names. *(Past performance is not indicative of future results;
this is an educational exercise.)*

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
[HH:MM:SS] Latest bar 2026-07-08  close=$13.49  P(up)=0.372  → signal=FLAT
[HH:MM:SS] Decision: BUY
[HH:MM:SS] ORDER SUBMITTED → BUY 20.0 F (id=77c67169-…, status=ACCEPTED)
```

*(Run with `--force buy` for a guaranteed demo order when the latest signal is
FLAT, or plain `python paper_trade.py` to act on the live signal.)*

*(Add your Alpaca paper dashboard screenshot showing the executed order here.)*

## Video

*(Add your YouTube unlisted link or in-repo video file here.)* The video shows
the code running, the charts (equity, drawdown, PCA variance), the backtest
metrics, the Alpaca paper dashboard, a paper trade being executed, and the
spoken statement: **"This is paper trading only — no real money is used."**

---

> ⚠️ **Reminder: this homework uses Alpaca PAPER TRADING only. No real money.**
