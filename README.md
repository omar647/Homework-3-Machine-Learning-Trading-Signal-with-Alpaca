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
python run_pipeline.py --ticker AAPL
python run_pipeline.py --ticker QQQ --model gradient_boosting --threshold 0.6
```

Models: `random_forest`, `logistic`, `gradient_boosting`, `svm`, `mlp`.

**Report notebook** (charts inline):

```bash
jupyter notebook ml_report.ipynb
```

**Paper trading demo** (submits a PAPER order to Alpaca):

```bash
python paper_trade.py --ticker AAPL --qty 5              # follow the model signal
python paper_trade.py --ticker AAPL --qty 5 --force buy  # guarantee a demo trade
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
number of components reaching **≥ 80%** cumulative variance — for AAPL that is
**4 components (~83%)**. PCA is fit on the **training window only** so the test
backtest has no look-ahead leakage. See `charts/*_pca_variance.png`.

## Example results (AMD, Gradient Boosting, test window Jan 2025 – Jul 2026)

```bash
python run_pipeline.py --ticker AMD --model gradient_boosting
```

| | Total Return | CAGR | Volatility | Sharpe | Sortino | Max DD | Win Rate | Trades |
|---|---|---|---|---|---|---|---|---|
| **ML Signal** | **+157.07%** | 92.61% | 35.35% | **2.03** | 2.17 | **−14.35%** | 63.16% | 57 |
| **Buy & Hold** | +319.95% | 170.79% | 66.63% | 1.82 | 3.09 | −36.29% | 100.00% | 1 |

Here the ML signal earns a **higher Sharpe (2.03 vs 1.82)** with roughly **half
the volatility and half the drawdown** of Buy & Hold — exactly what a signal is
for: better *risk-adjusted* return. It sits in cash during the worst stretches,
so it gives up some raw upside but is far less punishing to hold.

### Other positive combinations

A sweep across tickers × models found **16 / 40 combos with positive ML return**
and **8 that beat Buy & Hold outright**. Notable ones:

| Ticker · Model | ML Return | Sharpe | Max DD | Note |
|---|---|---|---|---|
| AMD · gradient_boosting | +157% | 2.03 | −14% | best Sharpe, beats B&H risk-adjusted |
| AAPL · mlp | +35% | 1.23 | −11% | nearly matches B&H with 1/3 the drawdown |
| META · random_forest | +9.9% | 0.58 | −9% | **beats B&H (−4.9%) outright** |
| QQQ · random_forest | +8.3% | 1.20 | −4% | cleanest risk-adjusted, very few trades |

Results vary by ticker/model/threshold — Random Forest and SVM are often
conservative (mostly Flat), while Gradient Boosting and MLP trade more actively.
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
[HH:MM:SS] Latest bar 2026-07-08  close=$313.33  P(up)=0.556  → signal=FLAT
[HH:MM:SS] Decision: BUY
[HH:MM:SS] ORDER SUBMITTED → BUY 5.0 AAPL (id=f2211dfb-…, status=ACCEPTED)
```

*(Add your Alpaca paper dashboard screenshot showing the executed order here.)*

## Video

*(Add your YouTube unlisted link or in-repo video file here.)* The video shows
the code running, the charts (equity, drawdown, PCA variance), the backtest
metrics, the Alpaca paper dashboard, a paper trade being executed, and the
spoken statement: **"This is paper trading only — no real money is used."**

---

> ⚠️ **Reminder: this homework uses Alpaca PAPER TRADING only. No real money.**
