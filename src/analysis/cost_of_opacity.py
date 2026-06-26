"""
Amplifier D, REFRAMED (Devil's Advocate II, CRITICAL-2).

The earlier framing compared Islamic products to a cheap conventional replica.
But that replica (ex-financials market + quality) is NOT Shariah compliant, so the
saving is illusory for a Muslim investor. Corrected framing: the cost of opacity
is measured WITHIN the compliant universe -- a transparent, passive, rules-based
Shariah ETF versus an active, opaque Shariah fund. The saving is real and halal.

Expense ratios are approximate published figures [VERIFY] before submission.
Run: python src/analysis/cost_of_opacity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "docs" / "results"

# all Shariah-COMPLIANT products [VERIFY expense ratios]
COMPLIANT = {
    "ISDU.L (iShares USA Islamic, passive)": (0.40, "passive/transparent"),
    "SPUS (SP Funds 500 Shariah, passive)":  (0.45, "passive/transparent"),
    "HLAL (Wahed FTSE USA Shariah, passive)": (0.50, "passive/transparent"),
    "UMMA (Wahed DJ Islamic World, passive)": (0.65, "passive/transparent"),
    "AMAGX (Amana Growth, active)":           (0.86, "active/opaque"),
    "AMANX (Amana Income, active)":           (1.00, "active/opaque"),
    "IMANX (Iman Fund, active)":              (1.30, "active/opaque"),
}

NOTIONAL = 10_000.0
GROSS = 0.08
YEARS = 10


def cost(er: float) -> float:
    return NOTIONAL * ((1 + GROSS) ** YEARS - (1 + GROSS - er / 100) ** YEARS)


def main() -> None:
    rows = [{"product": k, "type": t, "expense_%": er,
             f"{YEARS}yr_cost_$": round(cost(er))}
            for k, (er, t) in COMPLIANT.items()]
    df = pd.DataFrame(rows)
    print("=" * 80)
    print(f"Cost of opacity WITHIN the Shariah-compliant universe "
          f"(gross {GROSS:.0%}, {YEARS}y, ${NOTIONAL:,.0f})")
    print("=" * 80)
    print(df.to_string(index=False))

    passive = [er for er, t in COMPLIANT.values() if t == "passive/transparent"]
    active = [er for er, t in COMPLIANT.values() if t == "active/opaque"]
    p_lo, a_hi = min(passive), max(active)
    save_typical = cost(1.00) - cost(0.45)     # active Amana vs passive SPUS
    save_max = cost(a_hi) - cost(p_lo)
    print(f"\n  Compliant passive/transparent: {min(passive):.2f}-{max(passive):.2f}% ; "
          f"compliant active/opaque: {min(active):.2f}-{max(active):.2f}%.")
    print(f"  A transparent passive Shariah ETF vs an active opaque Shariah fund saves")
    print(f"  ~{round(100*(1.00-0.45))} bps/yr (SPUS vs AMANX) to ~{round(100*(a_hi-p_lo))} bps/yr,")
    print(f"  i.e. ~${save_typical:,.0f} to ~${save_max:,.0f} per ${NOTIONAL:,.0f} over {YEARS}y.")
    print("  Both sides are Shariah compliant: the premium is for opacity/active"
          "\n  management, not for compliance. (No non-compliant replica involved.)")
    df.to_csv(OUT / "cost_of_opacity.csv", index=False)


if __name__ == "__main__":
    main()
