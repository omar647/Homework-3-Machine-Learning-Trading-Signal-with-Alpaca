# 🎬 Video Script — Homework 3 (ML Trading Signal with Alpaca)

**Target length:** 3–6 minutes. Read the **SAY** lines out loud; do the **SHOW / DO**
actions on screen.

> **Note:** `ml_report.ipynb` and `run_pipeline.py` run the **same code** — both
> import from `src/`. The notebook just shows the charts inline, so it's the
> nicer thing to show on camera. The **live paper trade must be run in the
> terminal** (`paper_trade.py`), because that's what submits the Alpaca order.
> Plan = **notebook for the analysis + charts, terminal for the live trade.**

**Before you hit record:**
- Open the notebook in VS Code:  `code ml_report.ipynb`  (VS Code is installed;
  `jupyter` is not — so use VS Code, or `pip3 install jupyter` first).
- Open a terminal in the `homework3-ml-signal/` folder.
- Log in to your Alpaca **paper** dashboard in a browser tab
  (app.alpaca.markets → Paper).
- All commands use `python3` (on macOS `python` won't work).

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

**DO:** in VS Code, open the `src/` folder and scroll briefly through the files.

**SAY:**
> - "`data.py` pulls 5 years of daily OHLCV from Alpaca's Market Data API into a
>   pandas DataFrame — the ticker is chosen with a `--ticker` flag."
> - "`features.py` builds the features — SMA, EMA, MACD, ADX, RSI, Stochastic,
>   Williams %R, Bollinger Bands, ATR, OBV, CMF — plus log returns and rolling
>   mean and std."
> - "`pca.py` standardizes the features and keeps the components explaining at
>   least 80% of the variance."
> - "`model.py` trains the model. The target is next-day return greater than zero,
>   and it goes **Long if the probability is above 0.6, otherwise Flat.**"
> - "`backtest.py` is a long-only engine — $100k, no leverage, no shorting.
>   And the notebook I'm about to run imports all of these same files."

---

## 2 · Run the notebook — code + charts + results (1:10–3:20)

**DO:** open `ml_report.ipynb` in VS Code and click **Run All** (pick the Python 3
kernel if prompted). Scroll from the top as the cells execute.

**SAY (Data & Features cells):**
> "The notebook downloads 5 years of SPY data straight from Alpaca into a
> DataFrame, then builds the 29-column feature matrix you can see here."

**SAY (PCA cell — show the PCA variance chart):**
> "It standardizes the features and fits PCA. This chart shows each component's
> variance in bars, the cumulative variance as the red line, and the green line
> marks the **4 components we keep to cross the 80% threshold** — about 83% here.
> Those components are the model's inputs."

**SAY (Model / Signal cell — show the signal chart):**
> "Then Gradient Boosting is trained on those components. Up top is the price with
> green shading everywhere the model is **Long**; below is the model's probability
> of an up day, with the **0.6 threshold** line."

**SAY (Backtest cells — show equity + drawdown charts):**
> "Here's the backtest on a held-out test window the model never trained on — the
> **equity curve** of the ML signal versus Buy & Hold, both starting at $100k, and
> the **drawdown** chart underneath."

**SAY (Metrics table cell):**
> "And the metrics for both strategies — Total Return, CAGR, Volatility, Sharpe,
> Sortino, Max Drawdown, and Win Rate. Notice the ML signal holds a smaller
> drawdown because it steps to cash on weak days."

*(Read the actual numbers you see on screen.)*

---

## 3 · Alpaca paper dashboard + execute a trade (3:20–4:50)

**DO:** switch to the browser tab with your Alpaca **paper** dashboard.

**SAY:**
> "This is my Alpaca **paper trading** dashboard — you can see it says Paper and
> the account balance. Now I'll run the live trading script in the terminal."

**DO:** in the terminal, run the paper trader (`--force buy` guarantees a trade on
camera even if today's live signal is Flat):

```bash
python3 paper_trade.py --ticker SPY --qty 10 --force buy
```

**SHOW / SAY:** point at the log lines:
> "It connected to the paper account, fetched the latest data, applied PCA,
> generated the signal — and right here: **ORDER SUBMITTED, status ACCEPTED**,
> with the order ID. Buy if Long, sell if Flat — this one's a buy."

**DO:** switch back to the Alpaca dashboard and **refresh**. Show the order /
position.

**SAY:**
> "And here's that same order in the Alpaca dashboard — executed in the paper
> account."

---

## 4 · Close (4:50–5:10)

**SAY:**
> "So that's the full pipeline — Alpaca data, feature engineering, PCA, a Gradient
> Boosting signal, a backtest against Buy & Hold, and a live order in Alpaca.
> **Once more: this is paper trading only — no real money is used.** Thanks for
> watching."

---

## ✅ Checklist (the video must show all of these)

- [ ] Code running (the notebook executing, which runs the `src/` files)
- [ ] Charts: **equity curve, drawdown, PCA variance**
- [ ] Backtest results table (metrics)
- [ ] Alpaca **paper** dashboard
- [ ] A paper trade being executed (terminal log + order in dashboard)
- [ ] You saying: **"This is paper trading only — no real money is used."**

## Tips
- If the market is closed, the order status may be `ACCEPTED`/`PENDING` and fill
  at the next open — that still counts as "executed / submitted." You can also
  show a previously filled order in the dashboard's **Orders** history.
- Prefer the terminal instead of the notebook? `python3 run_pipeline.py --ticker SPY`
  prints the same metrics table and saves the same charts to `charts/` — it's the
  same code either way.
- Keep it tight: 3–6 minutes total. Summarize the code; don't read every line.
- Upload as **YouTube (unlisted)** and paste the link in the README, or drop the
  video file directly in the GitHub repo.
