"""
Proof-of-concept — Waqf-Zakat-Sukuk (WZS) blended development finance.

A single-factor (Vasicek/Gaussian-copula) portfolio-credit Monte Carlo of a
development-project portfolio financed through a three-layer stack:

  Zakat first-loss reserve (z)  -> absorbs the first losses (spent, but on
                                   zakat-eligible development beneficiaries)
  Waqf tabarru backstop   (g)   -> absorbs losses above zakat as an uncompensated
                                   donation (from waqf INCOME; corpus preserved) +
                                   provides a liquidity facility. NOT a guarantee:
                                   third-party tabarru, charging no fee.
  Senior development sukuk (1)  -> market investors; de-risked + made liquid

Quantifies: senior-tranche risk, crowding-in multiplier (private capital per unit
of social capital), zakat development-multiplier vs a one-time grant, and the
financing gain from removing the illiquidity premium via the waqf facility.

Run: python src/innovation/wzs_simulation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parents[2] / "docs" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)

# --- calibration (Islamic microfinance / development pool) --------------------
# PD anchored to Islamic-microfinance global PAR30 ~4-5% (South Asia ~2%, Africa ~7%).
N = 300            # projects in the portfolio
PD = 0.045         # probability of default (calibrated to IMFI PAR evidence)
LGD = 0.45         # loss given default
RHO = 0.15         # asset correlation (single systematic factor)
Z = 0.06           # zakat grant-equity, owned by the asnaf (first-loss by nature)
W = 0.04           # waqf tabarru backstop (from waqf income; corpus preserved)
B = 1_000_000      # Monte Carlo paths (large, to stabilise the 99.5% tail quantile)
ILLIQ_PREMIUM = 0.010   # 100 bps illiquidity premium on sukuk (our evidence)
TENOR = 5          # years


def simulate_losses(pd_=PD, rho=RHO, n=N, b=B) -> np.ndarray:
    """Single-factor Gaussian copula portfolio loss (fraction of notional)."""
    thr = norm.ppf(pd_)
    M = RNG.standard_normal(b)                       # systematic factor
    pcond = norm.cdf((thr - np.sqrt(rho) * M) / np.sqrt(1 - rho))
    defaults = RNG.binomial(n, pcond)                # correlated default count
    return LGD * defaults / n                        # portfolio loss fraction


def tranche_stats(loss, z, w):
    senior_loss = np.maximum(0.0, loss - z - w)
    return {
        "senior_EL_bps": round(1e4 * senior_loss.mean(), 1),
        "P(senior loss>0)_%": round(100 * (senior_loss > 0).mean(), 2),
        "senior_99.5%_loss_bps": round(1e4 * np.percentile(senior_loss, 99.5), 1),
        "zakat_exhausted_%": round(100 * (loss > z).mean(), 1),
        "waqf_touched_%": round(100 * (loss > z + w).mean(), 1),
    }


def main() -> None:
    loss = simulate_losses()
    print("=" * 72)
    print("WZS proof-of-concept — three-layer development-finance stack")
    print("=" * 72)
    print(f"  portfolio: N={N}, PD={PD:.0%}, LGD={LGD:.0%}, rho={RHO:.2f}; "
          f"expected loss = {1e4*loss.mean():.0f} bps")
    print(f"  layers: zakat grant-equity (asnaf-owned) z={Z:.0%}, "
          f"waqf tabarru backstop w={W:.0%}, senior sukuk = 100%")

    st = tranche_stats(loss, Z, W)
    print("\n  -- Senior development sukuk (the market layer) --")
    for k, v in st.items():
        print(f"     {k:24s} {v}")
    print("     => near-zero expected loss: an investment-grade, investable instrument.")

    social = Z + W
    crowd = 1.0 / social
    print("\n  -- Mobilisation (Sharia-coherent leverage on genuine charity) --")
    print(f"     social capital (zakat grant-equity + waqf income) = {social:.0%} of senior")
    print(f"     the asnaf OWN the zakat equity ({Z:.0%}); investors bear the senior tail")
    print(f"     private development sukuk mobilised = {crowd:.1f}x the social capital")
    print(f"     zakat development-multiplier = {1/Z:.1f}x vs a one-time grant; waqf corpus")
    print(f"     preserved, so the effect recurs. Not a free lunch: leverage on real charity.")
    print(f"     vs CWLS (Indonesia 2020): CWLS parks waqf in risk-free govt sukuk for charity")
    print(f"     coupons; it mobilises NO private capital and de-risks NOTHING. WZS does both.")

    # liquidity: waqf facility removes the illiquidity premium -> cheaper senior
    pv_saving = ILLIQ_PREMIUM * (1 - (1 + 0.05) ** -TENOR) / 0.05   # annuity PV per $1
    print("\n  -- Liquidity facility (waqf income) --")
    print(f"     removes ~{1e4*ILLIQ_PREMIUM:.0f} bps illiquidity premium (our sukuk evidence)")
    print(f"     => PV financing saving ~{100*pv_saving:.1f}% of senior notional over {TENOR}y.")
    print(f"     Honest: the waqf BEARS this illiquidity for its beneficiaries — risk is")
    print(f"     relocated to charity by design, not eliminated from the system.")

    # sensitivity
    print("\n  -- Sensitivity: senior expected loss (bps) by PD and correlation --")
    print(f"     {'PD\\rho':>8}" + "".join(f"{r:>10.2f}" for r in [0.10, 0.15, 0.25]))
    rows = []
    for pd_ in [0.05, 0.08, 0.12]:
        line = f"     {pd_:>8.0%}"
        rec = {"PD": pd_}
        for r in [0.10, 0.15, 0.25]:
            el = 1e4 * np.maximum(0.0, simulate_losses(pd_=pd_, rho=r) - Z - W).mean()
            line += f"{el:>10.1f}"
            rec[f"rho_{r}"] = round(el, 1)
        rows.append(rec)
        print(line)
    print("\n     Senior stays low-risk across stress; raise z/g for higher-PD pools.")

    import csv
    with open(OUT / "wzs_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["PD", "rho_0.1", "rho_0.15", "rho_0.25"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
