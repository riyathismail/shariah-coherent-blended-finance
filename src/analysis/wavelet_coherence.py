"""
Wavelet (scale-decomposition) coherence — frequency-domain confirmation.

The illiquidity thesis is frequency-dependent: if sukuk's comovement with
conventional fixed income lives at LOW frequencies (long horizons) and is weak at
HIGH frequencies (short horizons, where stale pricing dominates), that reinforces
the frequency-ladder result. Each return series is decomposed with a stationary
(undecimated) wavelet transform into frequency bands; the correlation between two
series' detail coefficients is reported by scale (short -> long horizon).

Run: python src/analysis/wavelet_coherence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pywt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "results"
LEVEL = 6
WAVELET = "db4"


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def scale_corr(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Correlation of detail coefficients by scale (fine -> coarse)."""
    n = len(a)
    L = 2 ** LEVEL
    pad = (-n) % L
    ap = np.concatenate([a, np.zeros(pad)])
    bp = np.concatenate([b, np.zeros(pad)])
    ca = pywt.swt(ap, WAVELET, level=LEVEL)   # [(cA_J,cD_J),...,(cA_1,cD_1)] coarse->fine
    cb = pywt.swt(bp, WAVELET, level=LEVEL)
    ca, cb = ca[::-1], cb[::-1]               # reverse -> fine (scale 1) -> coarse (scale J)
    cors = []
    for j in range(LEVEL):
        dA = ca[j][1][:n]
        dB = cb[j][1][:n]
        cors.append(round(float(np.corrcoef(dA, dB)[0, 1]), 2))
    return cors


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)

    pairs = [("SPSK", "AGG"), ("SPSK", "EMB"), ("SPUS", "SPY")]
    horizons = [f"~{2**(j+1)}d" for j in range(LEVEL)]
    rows = {}
    for a, b in pairs:
        d = r[[a, b]].dropna()
        rows[f"{a}-{b}"] = scale_corr(d[a].values, d[b].values)

    df = pd.DataFrame(rows, index=horizons).T
    print("=" * 64)
    print("Wavelet scale-decomposition correlation (fine -> coarse horizon)")
    print("=" * 64)
    print(f"  wavelet={WAVELET}, levels={LEVEL}, sample from {d.index.min().date()}")
    print(df.to_string())
    print("\n  SPSK-AGG / SPSK-EMB rising with horizon => sukuk comoves with")
    print("  conventional FI at LOW frequencies; high-frequency 'distinctiveness'")
    print("  is the stale-pricing band. SPUS-SPY high at all scales (equity = market).")
    df.to_csv(OUT / "wavelet_scale_corr.csv")


if __name__ == "__main__":
    main()
