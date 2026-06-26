"""
P4a-robust — gate the sukuk "distinct asset class" claim (R^2=0.25 daily).

Three checks:
  R1 WEEKLY decomposition  — if low R^2 survives weekly sampling, it is NOT just
       stale daily ETF pricing. Standard illiquidity remedy.
  R2 AMIHUD illiquidity    — is SPSK actually more illiquid than conventional
       bond ETFs? quantifies the stale-pricing worry directly.
  R3 AUGMENTED decomposition — add GCC equity (QAT,KSA), oil (Brent), USD (DXY):
       does the sukuk residual load on Gulf macro? what is left unexplained?

Run: python src/analysis/sukuk_robustness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
HAC = 5


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def reg(y, X):
    return sm.OLS(y, sm.add_constant(X), missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC})


def star(t):
    return "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""


def show(m, cols, label):
    print(f"\n{label}: n={int(m.nobs)}  R2={m.rsquared:.2f}  "
          f"alpha_ann%={100*252*m.params['const']:+.2f}(t={m.tvalues['const']:+.1f})")
    for c in cols:
        print(f"   {c:11s} {m.params[c]:+.3f} (t={m.tvalues[c]:+.1f}){star(m.tvalues[c])}")


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(PROC / "macro.csv", index_col=0, parse_dates=True)
    r = simple(prices)

    # ---- factor blocks ------------------------------------------------------
    fi = pd.DataFrame({
        "duration": r["IEF"],
        "ig_credit": r["LQD"] - r["IEF"],
        "high_yield": r["HYG"] - r["LQD"],
        "em_usd": r["EMB"] - r["LQD"],
        "green": r["BGRN"] - r["AGG"],
    })
    gcc = pd.DataFrame({
        "gcc": r[["QAT", "KSA"]].mean(axis=1),
        "oil": macro["brent"].reindex(r.index).pct_change(),
        "usd": r["DX-Y.NYB"],
    })

    # ===================================================== R1 weekly vs daily
    print("=" * 74)
    print("R1 — sukuk decomposition: DAILY vs WEEKLY (stale-pricing check)")
    print("=" * 74)
    daily = pd.concat([r["SPSK"].rename("sukuk"), fi], axis=1).dropna()
    md = reg(daily["sukuk"], daily[fi.columns])
    show(md, list(fi.columns), "DAILY")

    pw = prices.resample("W-FRI").last()
    rw = pw.apply(lambda s: s.dropna().pct_change())
    fiw = pd.DataFrame({
        "duration": rw["IEF"], "ig_credit": rw["LQD"] - rw["IEF"],
        "high_yield": rw["HYG"] - rw["LQD"], "em_usd": rw["EMB"] - rw["LQD"],
        "green": rw["BGRN"] - rw["AGG"]})
    wk = pd.concat([rw["SPSK"].rename("sukuk"), fiw], axis=1).dropna()
    mw = reg(wk["sukuk"], wk[fiw.columns])
    show(mw, list(fiw.columns), "WEEKLY")
    print(f"\n  Read: weekly R2={mw.rsquared:.2f} vs daily {md.rsquared:.2f}. If weekly stays"
          "\n  low => distinctiveness is real, not stale daily pricing.")

    # ===================================================== R2 Amihud illiquidity
    print("\n" + "=" * 74)
    print("R2 — Amihud illiquidity (x1e9): mean |ret| / dollar-volume")
    print("=" * 74)
    try:
        vol = pd.read_csv(PROC / "volume.csv", index_col=0, parse_dates=True)
        illiq = {}
        for a in ["SPSK", "AGG", "LQD", "EMB", "HYG", "SPY"]:
            if a in vol.columns and a in prices.columns:
                dollar = (prices[a] * vol[a]).replace(0, np.nan)
                ill = (r[a].abs() / dollar).replace([np.inf, -np.inf], np.nan).dropna()
                illiq[a] = round(1e9 * ill.mean(), 3)
        print(pd.Series(illiq, name="Amihud_x1e9").sort_values(ascending=False).to_string())
        print("  higher = more illiquid. If SPSK >> conventional bond ETFs, the low R2 is"
              "\n  partly illiquidity; if comparable, the distinctiveness is genuine.")
    except FileNotFoundError:
        print("  volume.csv not found — re-run pull_data.py to enable Amihud.")

    # ===================================================== R3 augmented (GCC/oil/USD)
    print("\n" + "=" * 74)
    print("R3 — augmented decomposition: + GCC equity / oil / USD")
    print("=" * 74)
    aug = pd.concat([r["SPSK"].rename("sukuk"), fi, gcc], axis=1).dropna()
    ma = reg(aug["sukuk"], aug[list(fi.columns) + list(gcc.columns)])
    show(ma, list(fi.columns) + list(gcc.columns), "AUGMENTED")
    print(f"\n  Read: R2 {md.rsquared:.2f} -> {ma.rsquared:.2f}. If GCC/oil/USD lift R2 a lot,"
          "\n  sukuk distinctiveness = Gulf macro exposure; if not, genuinely idiosyncratic"
          "\n  (asset-backed structure / local credit).")

    pd.DataFrame({"daily_R2": [md.rsquared], "weekly_R2": [mw.rsquared],
                  "augmented_R2": [ma.rsquared]}).to_csv(
        OUT / "sukuk_robustness_R2.csv", index=False)


if __name__ == "__main__":
    main()
