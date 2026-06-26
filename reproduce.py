"""
One-command reproduction of the entire IQT study from free public data.

This is amplifier C: the analysis is a fully open, auditable artifact. Anyone can
rebuild every number in the dossier with `python reproduce.py` (network needed for
the data pull). No proprietary data, no licensed index, no paywall.

    python reproduce.py            # full pipeline
    python reproduce.py --no-pull  # skip the data download, re-run analyses only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE = [
    ("data/pull_data.py", "Pull free secondary data (Yahoo, FRED, Ken French)"),
    ("analysis/descriptive.py", "Descriptive stats + regimes"),
    ("analysis/diagnostics.py", "Stationarity / normality / ARCH / VAR-lag diagnostics"),
    ("analysis/factor_attribution.py", "H0/H1/H2 factor attribution + matched benchmark"),
    ("analysis/event_study.py", "P3 event studies (double-dissociation)"),
    ("analysis/sukuk_analysis.py", "Sukuk decomposition + ballast"),
    ("analysis/sukuk_robustness.py", "Weekly + Amihud + GCC robustness"),
    ("analysis/asset_backed_control.py", "C1: conventional asset-backed control"),
    ("analysis/illiquidity_battery.py", "Illiquidity battery (frequency ladder, Dimson)"),
    ("analysis/inference_tests.py", "Bootstrap R2 CIs + equity-null robustness"),
    ("analysis/tail_systemic.py", "Tail dependence + CoVaR/MES + quantile beta"),
    ("analysis/dcc_hedge.py", "DCC-GARCH hedge ratios"),
    ("analysis/connectedness.py", "Diebold-Yilmaz connectedness"),
    ("analysis/markov_regime.py", "Markov-switching regimes"),
    ("analysis/iqt_index.py", "IQT index: replication + back-test"),
    ("analysis/cross_market.py", "Cross-market replication (amplifier B)"),
    ("analysis/cost_of_opacity.py", "Cost of opacity (amplifier D)"),
    ("analysis/wavelet_coherence.py", "Wavelet scale-decomposition coherence"),
    ("analysis/make_figures.py", "Generate paper figures (PNG)"),
]


def main() -> None:
    no_pull = "--no-pull" in sys.argv
    for rel, desc in PIPELINE:
        if no_pull and rel.startswith("data/"):
            print(f"[skip] {desc}")
            continue
        print(f"\n{'='*78}\n>>> {desc}\n    src/{rel}\n{'='*78}")
        rc = subprocess.run([sys.executable, str(ROOT / "src" / rel)]).returncode
        if rc != 0:
            sys.exit(f"FAILED at src/{rel} (exit {rc})")
    print("\nAll steps complete. Results in docs/results/.")


if __name__ == "__main__":
    main()
