"""
Generate the paper's figures (PNG) from processed data and result CSVs.
Run: python src/analysis/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RES = ROOT / "docs" / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

# Colourblind-safe palette; sukuk is highlighted in crimson across every figure
# so the eye tracks the same instrument throughout the paper.
C_SUKUK = "#d1495b"   # crimson — sukuk (SPSK)
C_EQUITY = "#2e7d32"  # green   — screened/Islamic equity
C_MARKET = "#1f6feb"  # blue    — market (SPY)
C_MBB = "#8338ec"     # violet  — agency MBS
C_CMBS = "#e8901a"    # amber   — commercial MBS
C_POS = "#c0392b"     # red     — net transmitter / worse tail
C_NEG = "#2a9d8f"     # teal    — net receiver / insulated


def fig_freq_ladder():
    df = pd.read_csv(RES / "illiquidity_battery.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = range(len(df.index))
    spec = [("MBB", "o", C_MBB, "MBB (agency MBS)", 1.8, 5),
            ("CMBS", "s", C_CMBS, "CMBS", 1.8, 5),
            ("SPSK", "^", C_SUKUK, "SPSK (sukuk)", 2.6, 8)]
    for col, mk, color, lab, lw, ms in spec:
        ax.plot(x, df[col], marker=mk, color=color, label=lab, lw=lw, markersize=ms)
    ax.set_xticks(list(x)); ax.set_xticklabels(df.index)
    ax.set_ylabel(r"factor $R^2$"); ax.set_ylim(0, 1)
    ax.set_title("Sukuk distinctiveness dissolves as stale pricing is removed")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "freq_ladder.png"); plt.close(fig)


def fig_cumret():
    px = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    cols = ["SPUS", "SPY", "SPSK"]
    d = px[cols].dropna()
    d = d / d.iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    style = {"SPUS": (C_EQUITY, "Islamic equity (SPUS)"),
             "SPY": (C_MARKET, "Market (SPY)"),
             "SPSK": (C_SUKUK, "Sukuk (SPSK)")}
    for c in cols:
        color, lab = style[c]
        ax.plot(d.index, d[c], color=color, label=lab, lw=1.6)
    ax.set_ylabel("cumulative growth (indexed to 1)"); ax.set_title("Cumulative total return")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "cumret.png"); plt.close(fig)


def fig_connectedness():
    df = pd.read_csv(RES / "connectedness.csv", index_col=0)
    net = df.loc["NET"].drop(labels=["FROM_others"], errors="ignore").dropna().astype(float)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    # teal = net receiver (insulated), red = net transmitter; sukuk kept crimson
    colors = [C_SUKUK if c == "SPSK" else (C_POS if v >= 0 else C_NEG)
              for c, v in net.items()]
    ax.bar(net.index, net.values, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("net directional connectedness")
    ax.set_title("Sukuk is the net receiver (most insulated)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "connectedness.png"); plt.close(fig)


def fig_mes():
    s = pd.read_csv(RES / "mes.csv", index_col=0)["MES"].sort_values()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    # red = worse than market on the worst days, teal = near zero / resilient
    colors = [C_SUKUK if c == "SPSK" else (C_POS if v < -1 else C_NEG)
              for c, v in s.items()]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean return on worst-5% market days (%/day)")
    ax.set_title("Sukuk near zero; screened equity worse than market")
    fig.tight_layout(); fig.savefig(FIG / "mes.png"); plt.close(fig)


def fig_wavelet():
    df = pd.read_csv(RES / "wavelet_scale_corr.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    palette = [C_SUKUK, C_MARKET, C_EQUITY, C_CMBS, C_MBB]
    for (pair, mk, color) in zip(df.index, ["^", "s", "o", "D", "v"], palette):
        ax.plot(df.columns, df.loc[pair].values, marker=mk, color=color,
                label=pair, lw=1.8)
    ax.set_ylabel("wavelet detail correlation"); ax.set_ylim(0, 1)
    ax.set_xlabel("timescale (short $\\to$ long horizon)")
    ax.set_title("Sukuk comoves with bonds only at low frequency")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "wavelet.png"); plt.close(fig)


def main():
    fig_freq_ladder(); fig_cumret(); fig_connectedness(); fig_mes(); fig_wavelet()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
