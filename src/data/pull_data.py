"""
Pull free secondary market data for the IQT Index / Concept C study.

Sources (all free):
  - Yahoo Finance (yfinance): daily OHLCV for Islamic + conventional + factor ETFs
  - Ken French Data Library (pandas_datareader): FF5 + momentum daily factors
  - FRED (pandas_datareader): rates, VIX, financial-conditions index

Outputs:
  data/raw/<ticker>.csv         per-ticker adjusted OHLCV
  data/processed/prices.csv     wide adjusted-close panel
  data/processed/returns.csv    daily log returns
  data/processed/ff_factors.csv FF5 + MOM (daily, decimal)
  data/processed/macro.csv      FRED controls

Run:  python src/data/pull_data.py
Deps: see requirements.txt  (yfinance, pandas, pandas_datareader)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------

START = "1999-01-01"   # long range: captures dot-com, GFC, COVID, AI regimes
END = None             # None = today
MAX_DAILY_LOGRET = 0.6  # |daily log return| above this = data error, masked

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# Ticker groups. The script downloads what exists and logs what fails,
# so unavailable/niche symbols never break the run.
TICKERS: dict[str, list[str]] = {
    # Shariah-compliant equity ETFs (tradable, free on Yahoo)
    "islamic_equity": ["SPUS", "HLAL", "UMMA"],
    # Long-history Shariah MUTUAL FUNDS (active) — robustness across regimes
    "islamic_long": ["AMANX", "AMAGX", "IMANX", "AMDEX"],
    # Shariah sukuk (fixed-income / sustainability sleeve proxy)
    "sukuk": ["SPSK"],
    # Conventional fixed-income factors to DECOMPOSE sukuk against + green proxy
    "fixed_income": ["AGG", "LQD", "HYG", "EMB", "TLT", "IEF", "BGRN"],
    # Conventional ASSET-BACKED / securitized controls (C1: is distinctiveness
    # asset-backing in general, or sukuk-specific?)
    "asset_backed": ["MBB", "VMBS", "CMBS"],
    # iShares MSCI Islamic UCITS (passive, inception 2007; clean mechanism proxy)
    "islamic_intl": ["ISDU.L", "ISDE.L", "ISWD.L"],
    # Conventional benchmarks (incl long fund/index for fund-vs-fund compare)
    "benchmark": ["SPY", "ACWI", "URTH"],
    "benchmark_long": ["VFINX", "^GSPC"],
    # GCC + macro: explain sukuk residual + Gulf relevance (Qatar/Saudi, USD)
    "gcc_macro": ["QAT", "KSA", "DX-Y.NYB"],
    # International benchmarks for cross-market replication (amplifier B)
    "intl_bench": ["EEM", "EFA"],
    # Factor / style ETFs
    "factor": ["QUAL", "MTUM", "USMV", "VLUE"],
    # Technology / AI exposure proxies
    "tech_ai": ["QQQ", "XLK", "SMH", "MAGS"],
    # Financials (the screened-out sector — for the no-bank test)
    "financials": ["XLF"],
    # Gold token (digital extension)
    "digital": ["PAXG-USD"],
}

# FRED controls + mechanism variables (must-adds from FT50 design review)
FRED_SERIES = {
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "VIXCLS": "vix",
    "NFCI": "nfci",
    "BAMLH0A0HYM2": "hy_oas",   # credit-stress mediator (credit leg)
    "DFF": "fed_funds",         # rate regime (hiking vs cutting)
    "DCOILBRENTEU": "brent",    # GCC/Qatar real-economy link (award fit)
    "USREC": "nber",            # objective recession regime
    "USEPUINDXD": "epu_us",     # economic policy uncertainty (daily)
}
# GPR (geopolitical risk, Caldara-Iacoviello) daily is an .xls manual download:
#   https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
# Not auto-fetched (xls + no stable CSV). Add manually to data/processed/ if needed.


# --- helpers ----------------------------------------------------------------

def _ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)


def fetch_yahoo() -> pd.DataFrame:
    """Download per-ticker OHLCV, save raw, return wide adjusted-close panel."""
    import yfinance as yf

    all_tickers = [t for group in TICKERS.values() for t in group]
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    failed: list[str] = []

    for t in all_tickers:
        try:
            df = yf.download(
                t, start=START, end=END,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or df.empty:
                failed.append(t)
                continue
            # yfinance may return a column MultiIndex for single tickers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(RAW / f"{t.replace('.', '_')}.csv")
            closes[t] = df["Close"].rename(t)
            if "Volume" in df.columns:
                volumes[t] = df["Volume"].rename(t)   # for Amihud illiquidity
            print(f"  ok   {t:10s} {df.index.min().date()} -> {df.index.max().date()}  n={len(df)}")
            time.sleep(0.3)  # be polite to the endpoint
        except Exception as e:  # noqa: BLE001 - log and continue
            failed.append(t)
            print(f"  FAIL {t:10s} {e}")

    if failed:
        print(f"\n[warn] {len(failed)} tickers unavailable, skipped: {failed}")

    if not closes:
        sys.exit("No tickers downloaded — check network / yfinance install.")

    prices = pd.concat(closes.values(), axis=1).sort_index()
    vol = pd.concat(volumes.values(), axis=1).sort_index() if volumes else pd.DataFrame()
    return prices, vol


def fetch_fama_french() -> pd.DataFrame:
    """FF5 + momentum daily factors, decimal (not percent).

    Downloads the CSV zips directly from the Ken French Data Library and parses
    the daily block by hand. This avoids pandas_datareader's PeriodIndex path,
    which breaks on newer pandas.
    """
    import io
    import urllib.request
    import zipfile

    base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

    def _load(zipname: str, cols: list[str]) -> pd.DataFrame:
        url = base + zipname
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows = []
        for line in raw:
            parts = [p.strip() for p in line.split(",")]
            # daily data rows start with an 8-digit YYYYMMDD date
            if parts and len(parts[0]) == 8 and parts[0].isdigit():
                rows.append(parts)
        df = pd.DataFrame(rows).set_index(0)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df.apply(pd.to_numeric, errors="coerce")
        df.columns = cols
        return df

    ff5 = _load("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _load("F-F_Momentum_Factor_daily_CSV.zip", ["MOM"])
    fac = ff5.join(mom, how="inner") / 100.0  # Ken French files are in percent
    return fac[fac.index >= pd.Timestamp(START)]


def fetch_fred() -> pd.DataFrame:
    """FRED macro controls."""
    from pandas_datareader.data import DataReader

    frames = []
    for code, name in FRED_SERIES.items():
        try:
            s = DataReader(code, "fred", start=START).rename(columns={code: name})
            frames.append(s)
        except Exception as e:  # noqa: BLE001
            print(f"  FRED FAIL {code}: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


# --- main -------------------------------------------------------------------

def main() -> None:
    _ensure_dirs()

    print("[1/3] Yahoo Finance ...")
    prices, vol = fetch_yahoo()
    prices.to_csv(PROC / "prices.csv")
    if not vol.empty:
        vol.to_csv(PROC / "volume.csv")
    # Per-column returns on each series' OWN calendar. Differencing the whole
    # panel would inject NaNs from PAXG's weekend rows and kill every equity
    # Monday-after-weekend. Compute on dropna(), then realign to the union index.
    returns = prices.apply(lambda s: np.log(s.dropna()).diff()).dropna(how="all")
    # Mask impossible daily moves (e.g. ISWD.L spurious ticks). Real equity/fund
    # days never exceed |0.6| in log terms; crypto stays well under it too.
    n_masked = int((returns.abs() > MAX_DAILY_LOGRET).sum().sum())
    returns = returns.mask(returns.abs() > MAX_DAILY_LOGRET)
    if n_masked:
        print(f"  [clean] masked {n_masked} outlier returns (|logret| > {MAX_DAILY_LOGRET})")
    returns.to_csv(PROC / "returns.csv")
    print(f"  prices  {prices.shape}  ->  data/processed/prices.csv")
    print(f"  returns {returns.shape}  ->  data/processed/returns.csv")

    print("[2/3] Fama-French factors ...")
    try:
        fac = fetch_fama_french()
        fac.to_csv(PROC / "ff_factors.csv")
        print(f"  factors {fac.shape}  ->  data/processed/ff_factors.csv")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Fama-French pull failed: {e}")

    print("[3/3] FRED controls ...")
    macro = fetch_fred()
    if not macro.empty:
        macro.to_csv(PROC / "macro.csv")
        print(f"  macro   {macro.shape}  ->  data/processed/macro.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
