"""Evaluation: AUPRC-first (the correct metric at 0.172% prevalence) + plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

DARK = "#0b0f1a"
GRID = "#26304a"
CYAN = "#22d3ee"
PURPLE = "#a78bfa"
RED = "#f87171"
GREEN = "#34d399"
TEXT = "#cbd5e1"


def _style(ax, title):
    ax.set_facecolor(DARK)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_title(title, color="white", fontsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def auprc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str):
    """curves: name -> (y_true, scores)"""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=DARK)
    palette = [CYAN, PURPLE, GREEN, RED]
    for (name, (y, s)), color in zip(curves.items(), palette):
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name}  (AUPRC {average_precision_score(y, s):.4f})")
    _style(ax, title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    leg = ax.legend(facecolor=DARK, edgecolor=GRID, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK)
    plt.close(fig)


def plot_latency_hist(latencies_ms: np.ndarray, budget_ms: float, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=DARK)
    ax.hist(latencies_ms, bins=40, color=CYAN, alpha=0.85)
    ax.axvline(budget_ms, color=RED, linestyle="--", linewidth=1.5,
               label=f"{budget_ms:.0f} ms sieve budget")
    _style(ax, "Synchronous Sieve — single-transaction latency")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("transactions")
    leg = ax.legend(facecolor=DARK, edgecolor=GRID, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK)
    plt.close(fig)


def plot_band_shift(p_sieve, final, y, lo, hi, path: Path):
    """How the Quantum Adjudicator moves scores inside the ambiguity band."""
    mask = (p_sieve >= lo) & (p_sieve <= hi)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=DARK)
    for label, color, name in [(0, CYAN, "legit"), (1, RED, "fraud")]:
        sel = mask & (y == label)
        ax.scatter(p_sieve[sel], final[sel], s=14, alpha=0.7, color=color, label=name)
    ax.plot([lo, hi], [lo, hi], color=TEXT, linewidth=0.8, linestyle=":")
    _style(ax, "Ambiguity band: sieve score -> mesh score (quantum-adjudicated)")
    ax.set_xlabel("sieve P(fraud)")
    ax.set_ylabel("mesh final score")
    leg = ax.legend(facecolor=DARK, edgecolor=GRID, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK)
    plt.close(fig)


def plot_loss(history: list[float], path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=DARK)
    ax.plot(range(1, len(history) + 1), history, color=PURPLE, linewidth=2)
    _style(ax, "Quantum Adjudicator dual loss  L = a*L_VQC + (1-a)*R_QAE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK)
    plt.close(fig)


def save_metrics(metrics: dict, path: Path):
    path.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics] written to {path}")
