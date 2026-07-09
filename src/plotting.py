"""Charts for the ML-signal homework.

  * PCA explained-variance (scree + cumulative)
  * Equity curves: ML signal vs Buy & Hold
  * Drawdowns
  * Price with model long/flat shading and the probability track
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .metrics import drawdown_series
from .pca import PCAModel


def plot_pca_variance(pca: PCAModel, save: str | None = None):
    """Scree bars + cumulative variance line, marking the 80% cutoff."""
    ratios = pca.explained_variance_ratio
    cum = np.cumsum(ratios)
    x = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, ratios * 100, alpha=0.6, color="#4c72b0", label="Individual")
    ax.plot(x, cum * 100, color="#c44e52", marker="o", label="Cumulative")
    ax.axhline(80, color="gray", linestyle="--", alpha=0.7, label="80% target")
    ax.axvline(pca.n_components, color="green", linestyle=":", alpha=0.8,
               label=f"{pca.n_components} components kept")
    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig


def plot_equity_curves(results: dict[str, BacktestResult], save: str | None = None):
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity, label=name, linewidth=1.6)
    ax.set_title("Equity Curve — ML Signal vs Buy & Hold (test window)")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig


def plot_drawdowns(results: dict[str, BacktestResult], save: str | None = None):
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, res in results.items():
        dd = drawdown_series(res.equity) * 100
        ax.plot(dd.index, dd, label=name, linewidth=1.3)
    ax.set_title("Drawdown — ML Signal vs Buy & Hold (test window)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig


def plot_signal(df_test: pd.DataFrame, proba: pd.Series, signal: pd.Series,
                threshold: float, title: str, save: str | None = None):
    """Price with green shading where the model is Long, plus the probability track."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    close = df_test["close"]
    ax1.plot(close.index, close, color="black", linewidth=1, label="Close")
    ax1.fill_between(close.index, close.min(), close.max(),
                     where=signal.reindex(close.index).fillna(0) == 1,
                     color="green", alpha=0.12, label="Model LONG")
    ax1.set_title(title)
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(proba.index, proba, color="#4c72b0", linewidth=1, label="P(up)")
    ax2.axhline(threshold, color="red", linestyle="--", alpha=0.7,
                label=f"threshold {threshold:.2f}")
    ax2.set_ylabel("P(up)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
    return fig
