"""
C1 — the identification control: asset-backing in general, or sukuk-specific?

Reviewer-2 objection: the equity(spanned)/sukuk(distinct) asymmetry is also an
equity/bond contrast. To attribute distinctiveness to ASSET-BACKING rather than
to "illiquid bond", test conventional asset-backed/securitized instruments
(agency MBS = MBB/VMBS, CMBS) against the same conventional factor set.

Logic:
  - If conventional asset-backed debt is ALSO poorly spanned (low R^2) -> the
    distinctiveness is a property of asset-backed/securitized debt in general,
    NOT Islamic. Reframe the headline.
  - If conventional asset-backed debt IS well spanned (high R^2) but sukuk is
    not -> a genuine sukuk-specific increment survives. Maqasid framework holds.
  - Then add an asset-backed factor (MBB) to the sukuk regression: does it absorb
    the sukuk residual?

Weekly returns (locked constraint: daily SPSK is illiquidity-contaminated).
Run: python src/analysis/asset_backed_control.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
HAC = 4


def reg(y, X):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC})
    return m, d


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    pw = prices.resample("W-FRI").last()
    rw = pw.apply(lambda s: s.dropna().pct_change())

    fi = pd.DataFrame({
        "duration": rw["IEF"],
        "ig_credit": rw["LQD"] - rw["IEF"],
        "high_yield": rw["HYG"] - rw["LQD"],
        "em_usd": rw["EMB"] - rw["LQD"],
        "green": rw["BGRN"] - rw["AGG"],
    })

    print("=" * 74)
    print("C1 — weekly R^2 on conventional FI factors: who is 'distinct'?")
    print("=" * 74)
    rows = {}
    for a in ["SPSK", "MBB", "VMBS", "CMBS", "LQD", "AGG", "EMB"]:
        if a not in rw.columns:
            continue
        m, d = reg(rw[a], fi)
        rows[a] = {"n": int(m.nobs), "R2": round(m.rsquared, 2),
                   "unexplained_%": round(100 * (1 - m.rsquared)),
                   "alpha_ann%": round(100 * 52 * m.params["const"], 1)}
    tbl = pd.DataFrame(rows).T
    print(tbl.to_string())
    print("\n  conventional asset-backed = MBB/VMBS/CMBS; plain bond control = LQD/AGG.")
    print("  If MBB/VMBS/CMBS are well spanned (high R2) but SPSK is not => sukuk-specific.")
    print("  If they are ALSO poorly spanned => distinctiveness is asset-backed debt generally.")
    tbl.to_csv(OUT / "c1_asset_backed_control.csv")

    # ---- does an asset-backed factor absorb the sukuk residual? --------------
    print("\n" + "=" * 74)
    print("C1b — add a conventional asset-backed factor (MBB) to the sukuk regression")
    print("=" * 74)
    base, _ = reg(rw["SPSK"], fi)
    fi_ab = fi.copy()
    fi_ab["mbs"] = rw["MBB"] - rw["IEF"]      # MBS excess over duration
    aug, d = reg(rw["SPSK"], fi_ab)
    print(f"  SPSK base R2 = {base.rsquared:.2f};  + MBS factor R2 = {aug.rsquared:.2f}  "
          f"(dR2 = {aug.rsquared-base.rsquared:+.2f})")
    print(f"  MBS loading = {aug.params['mbs']:+.3f} (t={aug.tvalues['mbs']:+.1f})")
    print("  Large dR2 / significant MBS loading => sukuk's residual IS conventional"
          "\n  asset-backed risk (not Islamic). Small => sukuk-specific component survives.")

    # ship the C1b absorption test (the increment claim's source artifact)
    pd.DataFrame({"base_R2": [round(base.rsquared, 4)],
                  "plus_mbs_R2": [round(aug.rsquared, 4)],
                  "dR2": [round(aug.rsquared - base.rsquared, 4)],
                  "mbs_loading": [round(aug.params["mbs"], 4)],
                  "mbs_t": [round(aug.tvalues["mbs"], 2)],
                  "n": [int(aug.nobs)]}).to_csv(
        OUT / "c1b_mbs_absorption.csv", index=False)


if __name__ == "__main__":
    main()
