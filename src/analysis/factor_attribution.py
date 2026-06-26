"""
P2 — Factor attribution for Concept C, conformed to the FT50 referee report.

Tests:
  H0 (CRITICAL): Islamic vs a MATCHED ex-financials benchmark. If the Islamic
       return does not differ from "the market with banks removed", the effect is
       not Shariah-specific. This is the anti-tautology test.
  H1: FF5 + MOM loadings (the tilt). HAC (Newey-West) SEs.
  H2 (OOS): betas estimated on train (<=2019), alpha evaluated on 2020-26 holdout.
  Mechanism: exposure to low-beta (USMV-SPY), credit (d HY-OAS), duration (d 10y).

Returns are SIMPLE (pct_change) to match Ken French factors (also simple).
Run: python src/analysis/factor_attribution.py
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

FF = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
ISLAMIC = ["AMANX", "AMAGX", "SPUS", "HLAL"]   # active funds (long) + passive ETFs
TRAIN_END = "2019-12-31"
FIN_WEIGHT = 0.13   # approx S&P 500 financials weight for ex-financials benchmark
HAC_LAGS = 5


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-column simple returns on each series' own calendar, outliers masked."""
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def hac(y: pd.Series, X: pd.DataFrame | None) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with Newey-West HAC SEs. X=None -> intercept-only (mean test)."""
    XX = sm.add_constant(X) if X is not None else pd.DataFrame(
        {"const": np.ones(len(y))}, index=y.index)
    return sm.OLS(y, XX, missing="drop").fit(cov_type="HAC",
                                             cov_kwds={"maxlags": HAC_LAGS})


def fmt(b: float, t: float) -> str:
    star = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""
    return f"{b:+.3f}({t:+.1f}){star}"


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    ff = pd.read_csv(PROC / "ff_factors.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(PROC / "macro.csv", index_col=0, parse_dates=True)

    r = simple_returns(prices)
    rf = ff["RF"]

    isl = [a for a in ISLAMIC if a in r.columns]

    # --- matched benchmark: S&P with financials stripped out -----------------
    ex_fin = ((r["SPY"] - FIN_WEIGHT * r["XLF"]) / (1 - FIN_WEIGHT)).rename("ex_fin")

    # ===================================================================== H1
    print("=" * 78)
    print("H1 — FF5 + MOM loadings (excess returns, HAC t-stats)  full sample")
    print("=" * 78)
    rows = {}
    for a in isl:
        df = pd.concat([r[a].rename("r"), ff[FF], rf.rename("rf")], axis=1).dropna()
        y = df["r"] - df["rf"]
        m = hac(y, df[FF])
        rows[a] = {"alpha_ann%": round(100 * 252 * m.params["const"], 1),
                   "alpha_t": round(m.tvalues["const"], 2),
                   "n": int(m.nobs), "R2": round(m.rsquared, 2)}
        rows[a].update({f: fmt(m.params[f], m.tvalues[f]) for f in FF})
    h1 = pd.DataFrame(rows).T
    print(h1.to_string())
    h1.to_csv(OUT / "h1_ff5_loadings.csv")

    # ===================================================================== H0
    print("\n" + "=" * 78)
    print("H0 — Islamic vs MATCHED ex-financials benchmark (anti-tautology)")
    print("    mean daily diff (bps) + HAC t; spanning alpha on ex_fin + QUAL + low-beta")
    print("=" * 78)
    low_beta = (r["USMV"] - r["SPY"]).rename("low_beta")
    qual = (r["QUAL"] - r["SPY"]).rename("qual")
    h0 = {}
    for a in isl:
        d = (r[a] - ex_fin).dropna()
        md = hac(d, None)
        # spanning: does ex_fin + style factors span Islamic? residual alpha = Shariah increment
        sp = pd.concat([r[a].rename("r"), ex_fin, qual, low_beta], axis=1).dropna()
        ms = hac(sp["r"], sp[["ex_fin", "qual", "low_beta"]])
        h0[a] = {
            "mean_diff_bps/day": round(1e4 * d.mean(), 2),
            "diff_t": round(md.tvalues["const"], 2),
            "span_alpha_ann%": round(100 * 252 * ms.params["const"], 2),
            "span_alpha_t": round(ms.tvalues["const"], 2),
            "beta_exfin": round(ms.params["ex_fin"], 2),
        }
    h0d = pd.DataFrame(h0).T
    print(h0d.to_string())
    print("\n  Read: span_alpha ~0 / insignificant  => effect IS 'ex-financials + style',"
          "\n        NOT Shariah-specific (honest). Significant +alpha => genuine increment.")
    h0d.to_csv(OUT / "h0_matched_benchmark.csv")

    # ===================================================================== H2
    print("\n" + "=" * 78)
    print(f"H2 — OUT-OF-SAMPLE alpha (betas trained <= {TRAIN_END}, tested 2020-26)")
    print("=" * 78)
    h2 = {}
    for a in isl:
        df = pd.concat([r[a].rename("r"), ff[FF], rf.rename("rf")], axis=1).dropna()
        y = df["r"] - df["rf"]
        tr = df.index <= TRAIN_END
        if tr.sum() < 250 or (~tr).sum() < 250:
            continue
        beta = sm.OLS(y[tr], sm.add_constant(df[FF])[tr]).fit().params
        pred = sm.add_constant(df[FF])[~tr] @ beta
        oos_alpha = (y[~tr] - pred)
        m = hac(oos_alpha, None)
        h2[a] = {"oos_alpha_ann%": round(100 * 252 * m.params["const"], 2),
                 "oos_t": round(m.tvalues["const"], 2),
                 "n_test": int(m.nobs)}
    h2d = pd.DataFrame(h2).T
    print(h2d.to_string())
    print("\n  Read: OOS alpha ~0 => outperformance is the tilt (H2 holds), not skill.")
    h2d.to_csv(OUT / "h2_oos_alpha.csv")

    # =============================================================== mechanism
    print("\n" + "=" * 78)
    print("MECHANISM — Islamic-minus-matched excess on low-beta / credit / duration")
    print("=" * 78)
    d_credit = macro["hy_oas"].diff().rename("d_credit")     # rising = stress
    d_rate = macro["ust_10y"].diff().rename("d_rate")        # rising = rate shock
    mech = {}
    for a in isl:
        df = pd.concat([(r[a] - ex_fin).rename("y"), low_beta, d_credit, d_rate],
                       axis=1).dropna()
        m = hac(df["y"], df[["low_beta", "d_credit", "d_rate"]])
        mech[a] = {k: fmt(m.params[k], m.tvalues[k])
                   for k in ["low_beta", "d_credit", "d_rate"]}
    print(pd.DataFrame(mech).T.to_string())
    print("\n  Expect: +credit loading (gains when spreads widen = bank-exclusion leg),"
          "\n          -duration loading (loses when rates rise = low-leverage/tech leg).")
    pd.DataFrame(mech).T.to_csv(OUT / "mechanism_exposures.csv")


if __name__ == "__main__":
    main()
