"""
Tier-2a — DCC-GARCH dynamic correlations + hedge ratios + hedging effectiveness.

Product question: for a Shariah equity book (SPUS), what is the best
Shariah-COMPLIANT hedge/ballast — sukuk (SPSK), vs conventional bond (AGG) or
gold (PAXG)? Hedge ratio beta_t = rho_t * h_i,t / h_j,t (time-varying, DCC).
Hedging effectiveness HE = 1 - Var(hedged)/Var(unhedged).

Univariate GJR-GARCH-t per series (arch) -> std resids -> DCC(1,1) by QMLE.
Run: python src/analysis/dcc_hedge.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"

ASSETS = ["SPUS", "SPSK", "AGG", "PAXG-USD", "SPY"]
PAIRS = [("SPUS", "SPSK"), ("SPUS", "AGG"), ("SPUS", "PAXG-USD"), ("SPUS", "SPY")]


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def dcc_correlations(Z: np.ndarray):
    """Estimate DCC(1,1) on standardized residuals; return per-t correlation matrices."""
    T, N = Z.shape
    Qbar = np.cov(Z.T)

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy()
        ll = 0.0
        for t in range(T):
            zt = Z[t]
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10
            ll += 0.5 * (logdet + zt @ np.linalg.solve(R, zt))
            Q = (1 - a - b) * Qbar + a * np.outer(zt, zt) + b * Q
        return ll

    opt = minimize(nll, [0.02, 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 400})
    a, b = opt.x
    Q = Qbar.copy()
    cors = np.empty((T, N, N))
    for t in range(T):
        zt = Z[t]
        d = np.sqrt(np.diag(Q))
        cors[t] = Q / np.outer(d, d)
        Q = (1 - a - b) * Qbar + a * np.outer(zt, zt) + b * Q
    return a, b, cors


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)[ASSETS].dropna()
    print(f"DCC sample {r.index.min().date()} -> {r.index.max().date()}  n={len(r)}")

    H, Z = {}, {}
    for a in ASSETS:
        res = arch_model(100 * r[a], mean="Constant", vol="GARCH",
                         p=1, o=1, q=1, dist="t").fit(disp="off")
        H[a] = res.conditional_volatility / 100.0   # back to return units
        Z[a] = res.std_resid
    H = pd.DataFrame(H).loc[r.index]
    Zdf = pd.DataFrame(Z).loc[r.index].dropna()
    r = r.loc[Zdf.index]
    H = H.loc[Zdf.index]

    a, b, cors = dcc_correlations(Zdf.values)
    idx = {name: k for k, name in enumerate(ASSETS)}
    print(f"DCC(1,1):  a={a:.3f}  b={b:.3f}  (a+b={a+b:.3f})\n")

    rows = {}
    for i, j in PAIRS:
        rho = pd.Series(cors[:, idx[i], idx[j]], index=Zdf.index)
        beta = rho * H[i] / H[j]
        hedged = r[i] - beta * r[j]
        he = 1 - hedged.var() / r[i].var()
        rows[f"{i} | {j}"] = {
            "avg_corr": round(rho.mean(), 2),
            "avg_hedge_ratio": round(beta.mean(), 2),
            "hedging_effectiveness_%": round(100 * he, 1),
        }
    tbl = pd.DataFrame(rows).T
    print(tbl.to_string())
    print("\n  Higher HE = better variance reduction. Compare sukuk(SPSK) as a"
          "\n  Shariah hedge vs conventional bond (AGG) and gold (PAXG).")
    tbl.to_csv(OUT / "dcc_hedge.csv")


if __name__ == "__main__":
    main()
