"""
Tier-2b / H4 — Diebold-Yilmaz (2012) generalized connectedness.

System: Islamic equity (SPUS), sukuk (SPSK), market (SPY), financials (XLF),
EM bonds (EMB), gold (PAXG). Generalized FEVD (Pesaran-Shin) at H=10 from a VAR.

Reports the connectedness matrix, TO / FROM / NET directional, and total
spillover. Key reads: is sukuk a NET receiver (insulated)? is Islamic equity
just the market (NET ~ market)?

Note: quantile/tail connectedness (Ando 2022) is a planned refinement; Tier-1
tail dependence already characterizes left-tail co-movement.
Run: python src/analysis/connectedness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"

SYSTEM = ["SPUS", "SPSK", "SPY", "XLF", "EMB", "PAXG-USD"]
H = 10
LAGS = 2


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def gfevd(ma, Sigma, H):
    """Generalized forecast error variance decomposition, row-normalized (%)."""
    N = Sigma.shape[0]
    theta = np.zeros((N, N))
    for i in range(N):
        denom = sum(ma[h][i] @ Sigma @ ma[h][i] for h in range(H))
        for j in range(N):
            num = sum((ma[h][i] @ Sigma[:, j]) ** 2 for h in range(H))
            theta[i, j] = (num / Sigma[j, j]) / denom
    theta = theta / theta.sum(axis=1, keepdims=True)   # row-normalize
    return 100 * theta


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)[SYSTEM].dropna()
    print(f"Connectedness sample {r.index.min().date()} -> {r.index.max().date()}  n={len(r)}")

    res = VAR(r).fit(LAGS)
    ma = res.ma_rep(maxn=H)            # (H+1, N, N), Phi_0 = I
    theta = gfevd(ma, res.sigma_u.values if hasattr(res.sigma_u, "values")
                  else np.asarray(res.sigma_u), H)

    N = len(SYSTEM)
    df = pd.DataFrame(theta, index=SYSTEM, columns=SYSTEM).round(1)
    frm = theta.sum(1) - np.diag(theta)           # FROM others
    to = theta.sum(0) - np.diag(theta)            # TO others
    net = to - frm
    df["FROM_others"] = frm.round(1)
    df.loc["TO_others"] = list(to.round(1)) + [np.nan]
    df.loc["NET"] = list(net.round(1)) + [np.nan]
    total = (theta.sum() - np.trace(theta)) / N
    print(f"\nTotal connectedness index: {total:.1f}%\n")
    print(df.to_string())
    print("\n  NET>0 = net transmitter; NET<0 = net receiver (insulated).")
    print("  Predict: SPSK strong net receiver; SPUS NET ~ SPY (Islamic eq = market).")
    df.to_csv(OUT / "connectedness.csv")


if __name__ == "__main__":
    main()
