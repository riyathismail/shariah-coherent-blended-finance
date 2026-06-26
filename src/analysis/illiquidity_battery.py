"""
Devil's Advocate II, CRITICAL-1 — is sukuk distinctiveness real, or stale pricing?

If the ~40% unexplained sukuk variation is an illiquidity/stale-price artifact, it
should DISappear as the sampling horizon lengthens (daily -> weekly -> monthly)
and as Dimson lags are added, converging toward the spanning of conventional
securitized debt (CMBS/MBS). If a substantial gap survives at monthly frequency
and after Dimson correction, the distinctiveness is not merely illiquidity.

Run: python src/analysis/illiquidity_battery.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"


def fi_factors(rw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "duration": rw["IEF"],
        "ig_credit": rw["LQD"] - rw["IEF"],
        "high_yield": rw["HYG"] - rw["LQD"],
        "em_usd": rw["EMB"] - rw["LQD"],
        "green": rw["BGRN"] - rw["AGG"],
    })


def r2(y, X):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 25:
        return np.nan, len(d)
    m = sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit()
    return m.rsquared, int(m.nobs)


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)

    print("=" * 70)
    print("Illiquidity battery — frequency ladder R^2 on conventional FI factors")
    print("=" * 70)
    rows = {}
    for freq, tag in [("D", "daily"), ("W-FRI", "weekly"), ("ME", "monthly")]:
        p = prices.resample(freq).last()
        rw = p.apply(lambda s: s.dropna().pct_change())
        fi = fi_factors(rw)
        rows[tag] = {}
        for a in ["SPSK", "CMBS", "MBB", "EMB"]:
            if a in rw.columns:
                val, n = r2(rw[a], fi)
                rows[tag][a] = round(val, 2)
        rows[tag]["n"] = n
    tbl = pd.DataFrame(rows).T
    print(tbl.to_string())
    print("\n  If SPSK R^2 climbs toward MBB/CMBS as horizon lengthens => the daily"
          "\n  'distinctiveness' was largely stale pricing.")

    # ---- Dimson correction at weekly ----------------------------------------
    print("\n" + "=" * 70)
    print("Dimson lags (weekly): SPSK ~ FI(t) + FI(t-1)  [+/- stale-price timing]")
    print("=" * 70)
    pw = prices.resample("W-FRI").last()
    rw = pw.apply(lambda s: s.dropna().pct_change())
    fi = fi_factors(rw)
    base, nb = r2(rw["SPSK"], fi)
    fi_lag = pd.concat([fi, fi.shift(1).add_suffix("_lag1")], axis=1)
    dim, nd = r2(rw["SPSK"], fi_lag)
    print(f"  base weekly R^2 = {base:.2f} (n={nb})")
    print(f"  + Dimson lag1   = {dim:.2f} (n={nd})   dR^2 = {dim-base:+.2f}")
    print("  Large dR^2 => contemporaneous regression understated spanning (stale prices).")

    # ship the Dimson result so the quoted 0.59->0.65 figure has a source artifact
    pd.DataFrame({"base_weekly_R2": [round(base, 4)],
                  "dimson_weekly_R2": [round(dim, 4)],
                  "dR2": [round(dim - base, 4)],
                  "n": [nd]}).to_csv(OUT / "dimson_weekly.csv", index=False)

    # verdict scaffold
    spsk_m = rows["monthly"].get("SPSK", np.nan)
    cmbs_m = rows["monthly"].get("CMBS", np.nan)
    print("\n" + "=" * 70)
    print(f"  VERDICT INPUT: SPSK monthly R^2 = {spsk_m}; CMBS monthly R^2 = {cmbs_m}; "
          f"Dimson-weekly R^2 = {dim:.2f}")
    print("  Survives (gap persists) => sukuk-specific. Converges => downgrade claim.")
    pd.DataFrame(rows).T.to_csv(OUT / "illiquidity_battery.csv")


if __name__ == "__main__":
    main()
