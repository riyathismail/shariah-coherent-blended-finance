# Shariah-Coherent Blended Finance — Replication Package

Code and data to reproduce the empirical analysis and the proof-of-concept
simulation behind the project's two manuscripts:

- **Concept paper** — *Shariah-Coherent Blended Finance: A Model of Zakat–Waqf–Sukuk
  Credit Enhancement for Real-Economy Development* (`paper/wzs-concept.pdf`).
- **Empirical study** — the illiquidity/spanning diagnosis of Shariah equity and
  sukuk that motivates it (`paper/main.pdf`).

Everything here runs from **free, public data** (Yahoo Finance, FRED, the Kenneth
French Data Library). No proprietary data is used.

---

## Data availability statement

The data that support the findings are openly available from public sources:
US-listed Shariah and conventional ETFs/funds and macro series from **Yahoo
Finance** and the **Federal Reserve Economic Data (FRED)** portal, and the
Fama–French factors from the **Kenneth R. French Data Library**. A processed
snapshot used by the analysis is included in `data/processed/`, and the analysis
outputs are included in `docs/results/`. All series are secondary and public;
the project introduces no new human-subjects or proprietary data. The complete
pipeline that pulls the raw data and regenerates every result is provided in this
repository under the MIT licence.

---

## How to reproduce

```bash
# 1) install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2) one command: pull free data, run all modules, write results + figures
python reproduce.py
```

`reproduce.py` writes the analysis outputs to `docs/results/*.csv` and the paper
figures to `paper/figures/*.png`. A committed snapshot of both is already in the
repository, so you can inspect the results without re-running. Re-running
regenerates them; the pipeline is deterministic and the bootstrap is seeded.

To run a single stage, execute any module directly, e.g.:

```bash
python src/analysis/illiquidity_battery.py     # sukuk illiquidity / frequency ladder
python src/innovation/wzs_simulation.py         # the WZS proof-of-concept simulation
```

---

## What is here

```
.
├── reproduce.py            # one-command pipeline (pull -> analyse -> figures)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT
├── src/
│   ├── data/pull_data.py   # downloads the free data into data/processed/
│   ├── analysis/           # 19 analysis modules (factor spanning, tails/systemic,
│   │                       #   illiquidity battery, connectedness, DCC, cost of
│   │                       #   opacity, cross-market, event study, etc.)
│   └── innovation/
│       └── wzs_simulation.py   # Vasicek single-factor PoC of the 3-layer facility
├── data/processed/         # processed input snapshot (prices, returns, FF factors, macro, volume)
├── docs/results/           # 26 result CSVs (the numbers reported in the papers)
└── paper/
    ├── figures/            # the 5 paper figures (PNG)
    ├── wzs-concept.pdf      # concept paper
    └── main.pdf             # empirical study
```

### Key result files

| File | Paper number it supports |
|---|---|
| `docs/results/h0_matched_benchmark.csv` | equity spanning null (H0) |
| `docs/results/illiquidity_battery.csv` | sukuk illiquidity frequency ladder |
| `docs/results/c1_asset_backed_control.csv`, `c1b_mbs_absorption.csv` | asset-backed control |
| `docs/results/wzs_sensitivity.csv` | WZS facility senior-EL sensitivity surface |
| `docs/results/connectedness.csv`, `mes.csv`, `dcovar.csv`, `tail_dependence.csv` | tails & systemic |

---

## Notes on the WZS proof-of-concept

`src/innovation/wzs_simulation.py` is an **illustrative, calibrated** single-factor
portfolio-credit simulation, not an empirical estimate. Its parameters (default
probability, loss given default, asset correlation) are calibrated to comparable
Islamic-microfinance evidence; the figures are model-conditional and are kept
distinct from the empirical results in `docs/results/`. The waqf layer is
third-party **tabarru** (uncompensated donation), not a guarantee.

---

## Licence & citation

Released under the **MIT licence** (see `LICENSE`). If you use this code or data,
please cite the corresponding manuscript(s); a `CITATION` entry will be added on
publication.
