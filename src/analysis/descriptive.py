"""
Descriptive statistics for the IQT / Concept C panel.

Reads data/processed/{prices,returns,ff_factors}.csv and reports, per ticker:
  sample range, n, annualized return, annualized vol, Sharpe (excess over RF),
  skew, excess kurtosis, max drawdown — full sample and the 2023+ AI-era subsample.
Also writes a correlation matrix for the key assets.

Run: python src/analysis/descriptive.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "descriptive"
OUT.mkdir(parents=True, exist_ok=True)

AI_ERA = "2023-01-01"
KEY = ["AMANX", "AMAGX", "IMANX", "ISWD.L", "SPUS", "HLAL", "UMMA",
       "VFINX", "SPY", "ACWI", "QQQ", "XLK", "QUAL", "XLF", "SPSK", "PAXG-USD"]

# Cross-regime robustness: long Shariah series (active funds + passive .L) vs
# conventional, across four stress regimes plus the AI era.
REGIMES = {
    "dotcom_2000_02": ("2000-03-01", "2002-10-31"),
    "GFC_2007_09": ("2007-10-01", "2009-03-31"),
    "covid_2020": ("2020-02-01", "2020-04-30"),
    "hiking_2022": ("2022-01-01", "2022-10-31"),
    "ai_2023_now": ("2023-01-01", None),
}
REGIME_COLS = ["AMANX", "AMAGX", "IMANX", "ISWD.L", "SPUS", "HLAL",
               "VFINX", "SPY", "QQQ", "XLK", "XLF", "SPSK"]


def max_drawdown(price: pd.Series) -> float:
    p = price.dropna()
    if p.empty:
        return np.nan
    return float((p / p.cummax() - 1.0).min())


def stats(ret: pd.Series, price: pd.Series, rf: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 30:
        return {}
    ann_ret = float(np.exp(252 * r.mean()) - 1)
    ann_vol = float(r.std() * np.sqrt(252))
    common = r.index.intersection(rf.index)
    rf_ann = float(252 * rf.loc[common].mean()) if len(common) else 0.0
    sharpe = (252 * r.mean() - rf_ann) / ann_vol if ann_vol else np.nan
    return {
        "n": len(r),
        "start": r.index.min().date(),
        "end": r.index.max().date(),
        "ann_ret_%": round(100 * ann_ret, 1),
        "ann_vol_%": round(100 * ann_vol, 1),
        "sharpe": round(sharpe, 2),
        "skew": round(float(r.skew()), 2),
        "exkurt": round(float(r.kurtosis()), 1),
        "maxDD_%": round(100 * max_drawdown(price), 1),
    }


def table(returns, prices, rf, cols, since=None) -> pd.DataFrame:
    if since:
        returns = returns.loc[since:]
        prices = prices.loc[since:]
        rf = rf.loc[since:]
    rows = {}
    for c in cols:
        if c in returns.columns:
            rows[c] = stats(returns[c], prices[c], rf)
    return pd.DataFrame(rows).T


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(PROC / "returns.csv", index_col=0, parse_dates=True)
    fac = pd.read_csv(PROC / "ff_factors.csv", index_col=0, parse_dates=True)
    rf = fac["RF"]

    cols = [c for c in KEY if c in returns.columns]

    full = table(returns, prices, rf, cols)
    ai = table(returns, prices, rf, cols, since=AI_ERA)

    full.to_csv(OUT / "stats_full.csv")
    ai.to_csv(OUT / "stats_ai_era.csv")

    corr = returns[cols].loc[AI_ERA:].corr().round(2)
    corr.to_csv(OUT / "corr_ai_era.csv")

    # Per-regime total return (%) — the cross-cycle robustness view
    reg_cols = [c for c in REGIME_COLS if c in returns.columns]
    reg_ret = {}
    for name, (a, b) in REGIMES.items():
        seg = returns.loc[a:b, reg_cols]
        # require >60% coverage in the window, else NaN (series not yet alive)
        tot = seg.apply(lambda s: 100 * (np.exp(s.dropna().sum()) - 1)
                        if s.notna().mean() > 0.6 else np.nan)
        reg_ret[name] = tot.round(1)
    reg_df = pd.DataFrame(reg_ret)
    reg_df.to_csv(OUT / "regime_total_return.csv")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== FULL SAMPLE (per-ticker own range) ===")
    print(full.to_string())
    print("\n=== AI ERA (2023-01-01 onward) ===")
    print(ai.to_string())
    print("\n=== CORRELATION (AI era, daily log returns) ===")
    print(corr.to_string())
    print("\n=== TOTAL RETURN % BY REGIME (NaN = series not yet alive) ===")
    print(reg_df.to_string())


if __name__ == "__main__":
    main()
