"""
P4a — Sukuk / sustainability pillar.

S1 (decomposition): what spans global sukuk (SPSK)? Regress on orthogonalized
    conventional fixed-income factors — duration (IEF), IG credit (LQD-IEF),
    high yield (HYG-LQD), EM USD (EMB-LQD), green (BGRN-AGG). HAC SEs.
    Honest analogue of the equity H0: find what sukuk economically IS.
S2 (ballast): does a sukuk sleeve diversify a Shariah equity book better than a
    conventional-bond sleeve? Compare 60/40 SPUS/SPSK vs 60/40 SPUS/AGG.
S3 (sustainability proxy): green vs conventional bonds (BGRN vs AGG).

Returns SIMPLE (pct_change). Run: python src/analysis/sukuk_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
OUT.mkdir(parents=True, exist_ok=True)
HAC_LAGS = 5


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def hac(y, X):
    XX = sm.add_constant(X)
    return sm.OLS(y, XX, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def fmt(b, t):
    star = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    return f"{b:+.3f}({t:+.1f}){star}"


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    ann = float(np.exp(np.log1p(r).sum() / len(r) * 252) - 1)
    vol = float(r.std() * np.sqrt(252))
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {"ann_ret_%": round(100 * ann, 1), "vol_%": round(100 * vol, 1),
            "sharpe": round(ann / vol, 2) if vol else np.nan, "maxDD_%": round(100 * dd, 1)}


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple_returns(prices)

    need = ["SPSK", "IEF", "LQD", "HYG", "EMB", "AGG", "BGRN", "SPUS", "SPY"]
    missing = [c for c in need if c not in r.columns]
    if missing:
        print(f"[warn] missing tickers: {missing}")

    # ----- S1: sukuk decomposition --------------------------------------------
    f = pd.DataFrame({
        "duration": r["IEF"],
        "ig_credit": r["LQD"] - r["IEF"],
        "high_yield": r["HYG"] - r["LQD"],
        "em_usd": r["EMB"] - r["LQD"],
        "green": r["BGRN"] - r["AGG"],
    })
    df = pd.concat([r["SPSK"].rename("sukuk"), f], axis=1).dropna()
    m = hac(df["sukuk"], df[list(f.columns)])
    print("=" * 80)
    print("S1 — What spans global sukuk (SPSK)?  HAC t-stats, simple returns")
    print("=" * 80)
    print(f"  sample {df.index.min().date()} -> {df.index.max().date()}  n={int(m.nobs)}  R2={m.rsquared:.2f}")
    print(f"  alpha (ann%): {100*252*m.params['const']:+.2f}  (t={m.tvalues['const']:+.2f})")
    for k in f.columns:
        print(f"  {k:11s} {fmt(m.params[k], m.tvalues[k])}")
    print("  Read: large duration+credit+EM betas, alpha~0 => sukuk is, economically,"
          "\n        an EM-tilted IG-credit/duration exposure (honest decomposition).")
    pd.Series({**{k: m.params[k] for k in f.columns},
               "alpha_ann%": 100*252*m.params['const'], "R2": m.rsquared}).to_csv(
        OUT / "s1_sukuk_decomposition.csv")

    # ----- S2: ballast diversification ---------------------------------------
    print("\n" + "=" * 80)
    print("S2 — Sukuk sleeve vs conventional-bond sleeve in a Shariah equity book")
    print("=" * 80)
    common = r[["SPUS", "SPSK", "AGG"]].dropna()
    blends = {
        "SPUS 100%": common["SPUS"],
        "60 SPUS / 40 SPSK (sukuk)": 0.6 * common["SPUS"] + 0.4 * common["SPSK"],
        "60 SPUS / 40 AGG (conv bond)": 0.6 * common["SPUS"] + 0.4 * common["AGG"],
    }
    tbl = pd.DataFrame({k: perf(v) for k, v in blends.items()}).T
    print(f"  sample {common.index.min().date()} -> {common.index.max().date()}  n={len(common)}")
    print(tbl.to_string())
    print(f"\n  corr(SPSK,SPUS)={common['SPSK'].corr(common['SPUS']):.2f}  "
          f"corr(SPSK,AGG)={common['SPSK'].corr(common['AGG']):.2f}  "
          f"corr(AGG,SPUS)={common['AGG'].corr(common['SPUS']):.2f}")
    tbl.to_csv(OUT / "s2_ballast.csv")

    # ----- S3: green vs conventional bonds -----------------------------------
    print("\n" + "=" * 80)
    print("S3 — Green vs conventional bonds (sustainability proxy)")
    print("=" * 80)
    g = r[["BGRN", "AGG"]].dropna()
    print(f"  sample {g.index.min().date()} -> {g.index.max().date()}  n={len(g)}")
    print(pd.DataFrame({"BGRN(green)": perf(g["BGRN"]), "AGG(conv)": perf(g["AGG"])}).T.to_string())
    print(f"  corr(BGRN,AGG)={g['BGRN'].corr(g['AGG']):.2f}")
    print("  Note: BGRN is a GREEN-BOND proxy, NOT green-sukuk specific — disclose"
          "\n        the proxy limitation; green-sukuk-specific free series are thin.")


if __name__ == "__main__":
    main()
