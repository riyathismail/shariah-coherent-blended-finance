"""
Preliminary analysis / diagnostics for the paper's methods section.

Descriptive statistics + stationarity (ADF, KPSS) + normality (Jarque-Bera) +
autocorrelation (Ljung-Box) + ARCH effects (Engle's LM) on the key return
series, and VAR lag selection for the connectedness system.

Run: python src/analysis/diagnostics.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"

SERIES = ["SPUS", "HLAL", "AMANX", "SPSK", "SPY", "XLF", "AGG", "EMB", "MBB", "CMBS"]
SYSTEM = ["SPUS", "SPSK", "SPY", "XLF", "EMB", "PAXG-USD"]


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def diag(x: pd.Series) -> dict:
    x = (x.dropna() * 100)   # percent returns
    adf = adfuller(x, autolag="AIC")
    kp = kpss(x, regression="c", nlags="auto")
    jb = jarque_bera(x)
    lb = acorr_ljungbox(x, lags=[10], return_df=True)
    arch = het_arch(x, nlags=10)
    return {
        "n": len(x),
        "mean": round(float(x.mean()), 3),
        "sd": round(float(x.std()), 2),
        "skew": round(float(x.skew()), 2),
        "ex_kurt": round(float(x.kurtosis()), 1),
        "min": round(float(x.min()), 1),
        "max": round(float(x.max()), 1),
        "ADF": round(float(adf[0]), 1),
        "ADF_p": round(float(adf[1]), 3),
        "KPSS": round(float(kp[0]), 2),
        "JB_p": round(float(jb[1]), 3),
        "LB10_p": round(float(lb["lb_pvalue"].iloc[0]), 3),
        "ARCH10_p": round(float(arch[1]), 3),
    }


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)

    rows = {a: diag(r[a]) for a in SERIES if a in r.columns}
    df = pd.DataFrame(rows).T
    print("=" * 100)
    print("Descriptive statistics and diagnostic tests (daily returns, %)")
    print("=" * 100)
    print(df.to_string())
    df.to_csv(OUT / "diagnostics.csv")

    print("\n  ADF_p < 0.05 => reject unit root (stationary). KPSS small => stationary.")
    print("  JB_p < 0.05 => non-normal. LB10_p < 0.05 => autocorrelation.")
    print("  ARCH10_p < 0.05 => ARCH effects present (justifies GARCH/DCC).")

    print("\n" + "=" * 60)
    print("VAR lag selection for the connectedness system")
    print("=" * 60)
    rv = r[SYSTEM].dropna()
    sel = VAR(rv).select_order(maxlags=10)
    print(f"  system: {SYSTEM}  n={len(rv)}")
    print(f"  selected lags: {sel.selected_orders}")


if __name__ == "__main__":
    main()
