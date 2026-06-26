"""
Tier-2c — Markov-switching regimes (endogenous calm/crisis, vs arbitrary dates).

2-regime Markov-switching on market returns with switching variance. Then check
how Islamic equity, financials, and sukuk behave in the data-defined crisis
regime. Answers Reviewer 2's "why these dates?" — the data pick the regimes.
Run: python src/analysis/markov_regime.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def simple(prices):
    r = prices.apply(lambda s: s.dropna().pct_change())
    return r.mask(r.abs() > 0.8)


def main() -> None:
    prices = pd.read_csv(PROC / "prices.csv", index_col=0, parse_dates=True)
    r = simple(prices)
    y = (100 * r["SPY"]).dropna()

    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True).fit()
    sig2 = mod.params[[p for p in mod.params.index if "sigma2" in p]].values
    crisis = int(np.argmax(sig2))   # high-variance regime
    print("=" * 64)
    print("Markov-switching (2 regimes, market SPY)")
    print("=" * 64)
    print(f"  regime variances: {np.round(sig2, 2)}  -> crisis regime = {crisis}")
    durs = 1 / (1 - np.diag(mod.regime_transition[:, :, 0]))
    print(f"  expected durations (days): calm={durs[1-crisis]:.0f}, crisis={durs[crisis]:.0f}")

    p_crisis = mod.smoothed_marginal_probabilities[crisis]
    in_crisis = p_crisis.reindex(r.index) > 0.5
    share = float(in_crisis.mean())
    print(f"  share of days in crisis regime: {100*share:.0f}%")

    print("\n  Mean return (%/day) by regime:")
    rows = {}
    for a in ["SPUS", "HLAL", "XLF", "SPSK", "SPY"]:
        if a in r.columns:
            rows[a] = {
                "calm": round(100 * r.loc[~in_crisis, a].mean(), 3),
                "crisis": round(100 * r.loc[in_crisis, a].mean(), 3),
            }
    print(pd.DataFrame(rows).T.to_string())
    print("\n  Sukuk (SPSK) should stay near zero in crisis while equities fall hard.")


if __name__ == "__main__":
    main()
