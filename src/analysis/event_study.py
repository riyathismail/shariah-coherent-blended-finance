"""
P3 — Event studies: the two-sided mechanism (double-dissociation).

For each dated event, cumulative abnormal return (CAR) of each Islamic series vs:
  - the MATCHED ex-financials benchmark  (ex_fin = (SPY - 0.13*XLF)/0.87)
  - raw SPY                              (for the "how much is just no-banks" gap)

Prediction (vs matched benchmark, so NOT tautological):
  credit-leg events (Lehman/SVB/CS/COVID) -> CAR > 0  (bank-exclusion + low-leverage)
  rate-leg  events (taper-2013/2018Q4/hiking) -> CAR < 0  (low-leverage/tech tilt hurts)
  ai events -> CAR > 0 (tech leg)

CAR = sum of daily abnormal returns over the window; t = CAR / (sd*sqrt(n)).
Pooling within a leg raises power vs single short windows.

Run: python src/analysis/event_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from config import EVENTS  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
OUT.mkdir(parents=True, exist_ok=True)

FIN_WEIGHT = 0.13
ISLAMIC = ["AMANX", "AMAGX", "IMANX", "ISDU.L", "SPUS", "HLAL", "UMMA"]
MIN_COVER = 0.6


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def car(abn: pd.Series) -> tuple[float, float, int]:
    a = abn.dropna()
    n = len(a)
    if n < 3:
        return np.nan, np.nan, n
    C = a.sum()
    se = a.std(ddof=1) * np.sqrt(n)
    t = C / se if se > 0 else np.nan
    return 100 * C, t, n


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple_returns(prices)
    ex_fin = (r["SPY"] - FIN_WEIGHT * r["XLF"]) / (1 - FIN_WEIGHT)
    isl = [a for a in ISLAMIC if a in r.columns]

    per_event = []
    leg_bucket: dict[str, list[float]] = {}

    for ev, (a0, b0, leg, sign) in EVENTS.items():
        win = (r.index >= a0) & (r.index <= b0)
        wlen = int(win.sum())
        if wlen < 3:
            continue
        for a in isl:
            seg = r.loc[win, a]
            if seg.notna().mean() < MIN_COVER:
                continue
            abn_ex = (r.loc[win, a] - ex_fin.loc[win])
            abn_spy = (r.loc[win, a] - r.loc[win, "SPY"])
            C_ex, t_ex, n = car(abn_ex)
            C_spy, _, _ = car(abn_spy)
            if np.isnan(C_ex):
                continue
            hit = (np.sign(C_ex) > 0) == (sign == "+")
            per_event.append({
                "event": ev, "leg": leg, "pred": sign, "series": a,
                "CAR_vs_exfin_%": round(C_ex, 2), "t": round(t_ex, 2),
                "CAR_vs_SPY_%": round(C_spy, 2), "n": n, "hit": hit,
            })
            leg_bucket.setdefault(leg, []).append(C_ex if sign == "+" else -C_ex)

    df = pd.DataFrame(per_event)
    df.to_csv(OUT / "event_study_car.csv", index=False)

    print("=" * 92)
    print("P3 EVENT STUDIES — CAR vs matched ex-financials benchmark (and vs raw SPY)")
    print("=" * 92)
    for leg in ["credit", "rate", "ai"]:
        sub = df[df.leg == leg]
        if sub.empty:
            continue
        print(f"\n--- {leg.upper()} leg (predict {'CAR>0' if leg!='rate' else 'CAR<0'}) ---")
        print(sub[["event", "series", "CAR_vs_exfin_%", "t",
                   "CAR_vs_SPY_%", "n", "hit"]].to_string(index=False))

    print("\n" + "=" * 92)
    print("DOUBLE-DISSOCIATION SUMMARY (sign-aligned: + = matches prediction)")
    print("=" * 92)
    summ = []
    for leg, vals in leg_bucket.items():
        v = np.array(vals)
        summ.append({
            "leg": leg, "n_obs": len(v),
            "mean_aligned_CAR_%": round(v.mean(), 2),
            "share_correct_sign": round((v > 0).mean(), 2),
            "t_pooled": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)
            if len(v) > 1 else np.nan,
        })
    sdf = pd.DataFrame(summ)
    print(sdf.to_string(index=False))
    sdf.to_csv(OUT / "event_study_legsummary.csv", index=False)
    print("\n  Aligned CAR = CAR if predicted '+', else -CAR. Positive mean + high"
          "\n  share_correct + |t|>2 => the mechanism fires in the predicted direction.")


if __name__ == "__main__":
    main()
