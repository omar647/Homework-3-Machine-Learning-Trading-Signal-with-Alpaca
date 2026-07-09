# 🎬 Video Script — Homework 3 (ML Trading Signal with Alpaca)

**Target length:** 3–6 minutes. Read the **SAY** lines out loud; do the **SHOW / DO**
actions on screen. All commands use `python3` (macOS).

**Before you hit record:**
- Open a terminal in the `homework3-ml-signal/` folder.
- Log in to your Alpaca **paper** dashboard in a browser tab (app.alpaca.markets → Paper).
- Have the `charts/` folder ready to open (Finder or an image viewer).
- Have `ml_report.ipynb` open (optional, looks great on camera).

---

## 0 · Intro & disclaimer (0:00–0:20)

**SAY:**
> "Hi, this is my Homework 3 — a machine-learning trading signal built on Alpaca
> market data. It downloads five years of data, engineers technical indicators,
> reduces them with PCA, trains a Gradient Boosting model to predict up days,
> backtests the signal, and then places a trade in Alpaca.
> **This is paper trading only — no real money is used.**"

**SHOW:** the project folder / README title on screen.

---

## 1 · Walk the code (0:20–1:10)

**DO:** open the `src/` folder and scroll briefly through these files.

**SAY:**
> - "`data.py` pulls 5 years of daily OHLCV from Alpaca's Market Data API into a
>   pandas DataFrame, and the ticker is chosen with a `--ticker` flag."
> - "`features.py` builds the feature matrix — SMA, EMA, MACD, ADX, RSI,
>   Stochastic, Williams %R, Bollinger Bands, ATR, OBV, CMF — plus log returns
>   and rolling mean and std."
> - "`pca.py` standardizes the features and keeps the components explaining at
>   least 80% of the variance."
> - "`model.py` trains the model. The target is: next-day return greater than
>   zero. It goes **Long if the model's probability is above 0.6, otherwise Flat.**"
> - "`backtest.py` is a long-only engine — $100k, no leverage, no shorting."

---

## 2 · Run the pipeline (1:10–2:10)

**DO:** run it (SPY is the default; you can name any ticker):

```bash
python3 run_pipeline.py --ticker SPY
```

**SAY while it runs:**
> "It's downloading the data, engineering the features, fitting PCA — you can see
> it kept 4 components for about 83% of the variance — training the Gradient
> Boosting model, and backtesting the signal against Buy & Hold on a held-out
> test window that the model never trained on."

**SHOW:** the printed **Performance Comparison** table. Read the key numbers:

**SAY:**
> "Here are the metrics for both strategies — Total Return, CAGR, Volatility,
> Sharpe, Sortino, Max Drawdown, and Win Rate. The ML signal has a noticeably
> smaller drawdown than Buy & Hold because it moves to cash on weak days."

*(Say the actual numbers you see on screen.)*

---

## 3 · Show the charts (2:10–3:10)

**DO:** open the `charts/` folder and show each PNG.

**SAY:**
> - "**PCA variance chart** — the bars are each component's variance, the red line
>   is the cumulative total, and the green line marks the 4 components we keep to
>   pass the 80% threshold."
> - "**Signal chart** — price on top with green shading everywhere the model is
>   Long, and the model's probability of an up day on the bottom with the 0.6
>   threshold line."
> - "**Equity curve** — the ML signal versus Buy & Hold, both starting at $100k."
> - "**Drawdown chart** — how far each strategy falls from its peak."

*(Optional: show the same charts inline in `ml_report.ipynb`.)*

---

## 4 · Alpaca paper dashboard + execute a trade (3:10–4:40)

**DO:** switch to the browser tab with your Alpaca **paper** dashboard.

**SAY:**
> "This is my Alpaca **paper trading** dashboard — you can see it says Paper, and
> the account balance. Now I'll run the live trading script. It fetches the latest
> data, rebuilds the features, applies PCA, generates the signal, and submits a
> **paper** order — buy if the signal is Long, sell if it's Flat."

**DO:** run the paper trader (use `--force buy` so a trade is guaranteed on camera):

```bash
python3 paper_trade.py --ticker SPY --qty 10 --force buy
```

**SHOW / SAY:** point at the log lines:
> "You can see it connected to the paper account, the latest bar and the model's
> signal, the decision, and — right here — **ORDER SUBMITTED, status ACCEPTED**,
> with the order ID."

**DO:** switch back to the Alpaca dashboard and **refresh**. Show the order/position.

**SAY:**
> "And here's that same order showing up in the Alpaca dashboard — the trade was
> executed in the paper account."

---

## 5 · Close (4:40–5:00)

**SAY:**
> "So that's the full pipeline — Alpaca data, feature engineering, PCA, a Gradient
> Boosting signal, a backtest against Buy & Hold, and a live order in Alpaca.
> **Once more: this is paper trading only — no real money is used.** Thanks for
> watching."

---

## ✅ Checklist (make sure the video shows all of these)

- [ ] Code running (`run_pipeline.py`)
- [ ] Charts: **equity curve, drawdown, PCA variance**
- [ ] Backtest results table (metrics)
- [ ] Alpaca **paper** dashboard
- [ ] A paper trade being executed (log + order in dashboard)
- [ ] You saying: **"This is paper trading only — no real money is used."**

## Tips
- If the market is closed, the order status may be `ACCEPTED`/`PENDING` and fill
  at the next open — that still counts as "executed / submitted." You can also
  show a previously filled order in the dashboard's **Orders** history.
- Keep it tight: 3–6 minutes total. Don't read every line of code — summarize.
- Upload as **YouTube (unlisted)** and paste the link in the README, or drop the
  video file directly in the GitHub repo.
