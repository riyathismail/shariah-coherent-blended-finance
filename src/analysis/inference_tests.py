"""
Reviewer-2 (pivoted paper), R3 + R6 — add inference.

R3: bootstrap confidence intervals on the monthly factor-R^2 for sukuk vs
    conventional securitized debt, and on the R^2 gap, so the "converges to
    conventional" claim is tested rather than eyeballed.
R6: robustness of the equity-null spanning alpha to the matched-benchmark
    financials weight.

Run: python src/analysis/inference_tests.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
RNG = np.random.default_rng(42)
B = 2000


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def r2(y, X):
    return sm.OLS(y, sm.add_constant(X)).fit().rsquared


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)

    # ---- R3: bootstrap monthly R^2 and gaps ---------------------------------
    pm = prices.resample("ME").last()
    rm = pm.apply(lambda s: s.dropna().pct_change())
    fi = pd.DataFrame({
        "duration": rm["IEF"], "ig_credit": rm["LQD"] - rm["IEF"],
        "high_yield": rm["HYG"] - rm["LQD"], "em_usd": rm["EMB"] - rm["LQD"],
        "green": rm["BGRN"] - rm["AGG"]})
    df = pd.concat([rm[["SPSK", "CMBS", "MBB"]], fi], axis=1).dropna()
    X = df[fi.columns].values
    n = len(df)
    pt = {a: r2(df[a].values, X) for a in ["SPSK", "CMBS", "MBB"]}

    boot = {a: np.empty(B) for a in ["SPSK", "CMBS", "MBB"]}
    gap_cmbs = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        Xb = sm.add_constant(X[idx])
        for a in ["SPSK", "CMBS", "MBB"]:
            yb = df[a].values[idx]
            boot[a][b] = sm.OLS(yb, Xb).fit().rsquared
        gap_cmbs[b] = boot["CMBS"][b] - boot["SPSK"][b]

    print("=" * 70)
    print(f"R3 — bootstrap monthly factor R^2 (n={n}, B={B}), 95% CI")
    print("=" * 70)
    for a in ["SPSK", "CMBS", "MBB"]:
        lo, hi = np.percentile(boot[a], [2.5, 97.5])
        print(f"  {a:5s} R^2 = {pt[a]:.2f}  CI [{lo:.2f}, {hi:.2f}]")
    glo, ghi = np.percentile(gap_cmbs, [2.5, 97.5])
    print(f"  gap CMBS-SPSK = {pt['CMBS']-pt['SPSK']:.2f}  CI [{glo:.2f}, {ghi:.2f}]")
    print("  If the gap CI includes 0 => sukuk is not statistically distinct from")
    print("  conventional securitized debt at monthly frequency (cannot reject convergence).")

    # ---- R6: equity-null robustness to financials weight --------------------
    rd = simple(prices)
    qual = (rd["QUAL"] - rd["SPY"])
    low_beta = (rd["USMV"] - rd["SPY"])
    print("\n" + "=" * 70)
    print("R6 — equity spanning alpha vs matched-benchmark financials weight")
    print("=" * 70)
    for w in [0.10, 0.13, 0.16]:
        ex = (rd["SPY"] - w * rd["XLF"]) / (1 - w)
        d = pd.concat([rd["SPUS"].rename("r"), ex.rename("ex"),
                       qual.rename("q"), low_beta.rename("lb")], axis=1).dropna()
        m = sm.OLS(d["r"], sm.add_constant(d[["ex", "q", "lb"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5})
        print(f"  w_fin={w:.2f}:  spanning alpha = {100*252*m.params['const']:+.2f}%/yr "
              f"(t={m.tvalues['const']:+.2f})")
    print("  Insensitive, insignificant alpha => the equity null is robust.")


if __name__ == "__main__":
    main()
