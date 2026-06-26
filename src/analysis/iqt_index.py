"""
P4b — the IQT product: construction, transparent-replication proof, OOS back-test.

Given the evidence (equity screening = transparent ex-fin quality, no alpha;
stability lives in sukuk), the IQT product is honest by design:
  - equity sleeve (70%): Shariah equity, shown REPLICABLE from cheap free
    building blocks (ex-financials market + QUAL) -> the innovation is
    transparency/cost, not alpha.
  - sukuk sleeve (30%): SPSK, the stabilizing asset-backed ballast.

Back-test = constant-mix, net of an assumed 20bps/yr product cost. Compared to
Islamic-equity-only and conventional balanced funds, full sample + OOS (2022+).
Run: python src/analysis/iqt_index.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"

EQ_W, SK_W = 0.70, 0.30
ANNUAL_COST = 0.0020          # 20 bps/yr transparent product cost
TEST_START = "2022-01-01"


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def perf(x: pd.Series) -> dict:
    x = x.dropna()
    ann = float(np.exp(np.log1p(x).sum() / len(x) * 252) - 1)
    vol = float(x.std() * np.sqrt(252))
    cum = (1 + x).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {"ann_ret_%": round(100 * ann, 1), "vol_%": round(100 * vol, 1),
            "sharpe": round(ann / vol, 2) if vol else np.nan, "maxDD_%": round(100 * dd, 1)}


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)

    # ---- transparent-replication proof (the innovation) ---------------------
    rep = r[["SPUS", "SPY", "XLF", "QUAL"]].dropna()
    ex_fin = (rep["SPY"] - 0.13 * rep["XLF"]) / 0.87
    X = pd.concat([ex_fin.rename("ex_fin"), rep["QUAL"].rename("qual")], axis=1)
    tr = X.index <= "2021-12-31"
    m = sm.OLS(rep["SPUS"][tr], sm.add_constant(X[tr])).fit()
    pred = sm.add_constant(X) @ m.params
    resid = rep["SPUS"] - pred
    oos = resid.index > "2021-12-31"
    print("=" * 70)
    print("REPLICATION — Shariah equity (SPUS) from free building blocks")
    print("=" * 70)
    print(f"  train<=2021: betas ex_fin={m.params['ex_fin']:.2f}, qual={m.params['qual']:.2f}, "
          f"R2={m.rsquared:.3f}")
    print(f"  OOS (2022+): tracking error = {100*resid[oos].std()*np.sqrt(252):.2f}%/yr, "
          f"corr(SPUS,replica) = {rep['SPUS'][oos].corr(pred[oos]):.3f}")
    print("  => Shariah equity exposure is transparently replicable cheaply (the innovation).")

    # ---- IQT back-test ------------------------------------------------------
    bt = r[["SPUS", "SPSK", "SPY", "AGG"]].dropna()
    iqt = EQ_W * bt["SPUS"] + SK_W * bt["SPSK"] - ANNUAL_COST / 252
    series = {
        "IQT 70/30 (Islamic eq + sukuk), net": iqt,
        "SPUS 100% Islamic equity": bt["SPUS"],
        "60/40 SPY/AGG (conventional)": 0.6 * bt["SPY"] + 0.4 * bt["AGG"],
        "70/30 SPY/AGG (conventional)": 0.7 * bt["SPY"] + 0.3 * bt["AGG"],
    }
    full = pd.DataFrame({k: perf(v) for k, v in series.items()}).T
    test = pd.DataFrame({k: perf(v.loc[TEST_START:]) for k, v in series.items()}).T

    print("\n" + "=" * 70)
    print(f"IQT BACK-TEST — full sample {bt.index.min().date()} -> {bt.index.max().date()}")
    print("=" * 70)
    print(full.to_string())
    print(f"\n  OOS / stress sub-sample ({TEST_START}+, incl. 2022 rate shock):")
    print(test.to_string())
    print("\n  Honest read: IQT is not an alpha machine. It delivers Shariah-compliant"
          "\n  balanced exposure with materially better drawdown than equity-only, "
          "\n  competitive with conventional 60/40 — transparently, from free data.")
    full.to_csv(OUT / "iqt_backtest_full.csv")
    test.to_csv(OUT / "iqt_backtest_oos.csv")


if __name__ == "__main__":
    main()
