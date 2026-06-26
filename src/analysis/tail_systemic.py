"""
P4b Tier-1 — tail dependence + systemic risk + quantile betas.

Tests the PRE-REGISTERED predictions (docs/plans, section 4c):
  P1 bank decoupling : lambda_L(Islamic, XLF) < lambda_L(SPY, XLF)
  P3 sukuk MES       : MES(sukuk) ~ 0 vs equities on worst-5% market days
  P4 sukuk tail dec. : lambda_L(SPSK, SPY) ~ 0
  P5 downside bank-beta compression : quantile beta to XLF lower at tau=0.05 than 0.50
  P6 systemic        : |dCoVaR(Islamic)| <= |dCoVaR(XLF)|

Nonparametric tail dependence (ECDF pseudo-obs). CoVaR via quantile regression
(Adrian-Brunnermeier). Returns SIMPLE. Run: python src/analysis/tail_systemic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def simple_returns(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def tail_dep(x: pd.Series, y: pd.Series, q: float, lower: bool) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 100:
        return np.nan
    u = d.iloc[:, 0].rank() / (n + 1)
    v = d.iloc[:, 1].rank() / (n + 1)
    if lower:
        joint = ((u <= q) & (v <= q)).mean()
    else:
        joint = ((u > 1 - q) & (v > 1 - q)).mean()
    return joint / q


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple_returns(prices)
    ISL = [c for c in ["SPUS", "HLAL", "AMANX"] if c in r.columns]

    # ---- P1 / P4: lower-tail dependence -------------------------------------
    print("=" * 76)
    print("P1/P4 — lower-tail dependence  lambda_L (q=0.05)   [crash co-movement]")
    print("=" * 76)
    rows = {}
    for a in ISL + ["SPY", "QQQ"]:
        rows[a] = {"lamL_vs_XLF": round(tail_dep(r[a], r["XLF"], 0.05, True), 3),
                   "lamL_vs_SPY": round(tail_dep(r[a], r["SPY"], 0.05, True), 3),
                   "lamU_vs_SPY": round(tail_dep(r[a], r["SPY"], 0.05, False), 3)}
    rows["SPSK(sukuk)"] = {
        "lamL_vs_XLF": round(tail_dep(r["SPSK"], r["XLF"], 0.05, True), 3),
        "lamL_vs_SPY": round(tail_dep(r["SPSK"], r["SPY"], 0.05, True), 3),
        "lamU_vs_SPY": round(tail_dep(r["SPSK"], r["SPY"], 0.05, False), 3)}
    td = pd.DataFrame(rows).T
    print(td.to_string())
    td.to_csv(OUT / "tail_dependence.csv")
    base = td.loc["SPY", "lamL_vs_XLF"]
    print(f"\n  P1: SPY-XLF lower-tail dep = {base}. Islamic < this => bank decoupling.")
    print(f"  P4: SPSK-SPY lower-tail dep = {td.loc['SPSK(sukuk)','lamL_vs_SPY']} (~0 => safe-haven).")

    # ---- P3: MES on worst-5% market days ------------------------------------
    print("\n" + "=" * 76)
    print("P3 — Marginal Expected Shortfall: mean return (%/day) on worst-5% SPY days")
    print("=" * 76)
    mkt = r["SPY"]
    bad = mkt <= mkt.quantile(0.05)
    mes = {}
    for a in ISL + ["SPY", "XLF", "QQQ", "SPSK", "AGG", "PAXG-USD"]:
        if a in r.columns:
            mes[a] = round(100 * r.loc[bad, a].mean(), 2)
    print(pd.Series(mes, name="MES_%/day").to_string())
    print("  less negative = more resilient. Predict sukuk(SPSK) ~ 0, equities deeply negative.")
    pd.Series(mes, name="MES").to_csv(OUT / "mes.csv")

    # ---- P6: dCoVaR (system = SPY) ------------------------------------------
    print("\n" + "=" * 76)
    print("P6 — dCoVaR contribution to SPY system (tau=0.05), %/day")
    print("=" * 76)
    dcov = {}
    for a in ISL + ["XLF", "SPSK"]:
        d = pd.concat([r["SPY"].rename("sys"), r[a].rename("a")], axis=1).dropna()
        m = QuantReg(d["sys"], np.column_stack([np.ones(len(d)), d["a"]])).fit(q=0.05)
        b = m.params.iloc[1]
        dcov[a] = round(100 * b * (d["a"].quantile(0.05) - d["a"].median()), 3)
    print(pd.Series(dcov, name="dCoVaR_%").to_string())
    print("  smaller |value| = contributes less to system tail. Predict Islamic < XLF.")
    pd.Series(dcov, name="dCoVaR").to_csv(OUT / "dcovar.csv")

    # ---- P5: quantile beta to XLF -------------------------------------------
    print("\n" + "=" * 76)
    print("P5 — quantile beta to financials (XLF) across tau  [downside compression?]")
    print("=" * 76)
    taus = [0.05, 0.25, 0.50, 0.75, 0.95]
    qb = {}
    for a in ISL:
        d = pd.concat([r[a].rename("a"), r["XLF"].rename("x")], axis=1).dropna()
        qb[a] = {f"b@{t:.2f}": round(
            QuantReg(d["a"], np.column_stack([np.ones(len(d)), d["x"]])).fit(q=t).params.iloc[1], 3)
            for t in taus}
    qbd = pd.DataFrame(qb).T
    print(qbd.to_string())
    print("  P5 holds if b@0.05 < b@0.50 (lower bank-beta in the left tail).")
    qbd.to_csv(OUT / "quantile_beta_xlf.csv")


if __name__ == "__main__":
    main()
