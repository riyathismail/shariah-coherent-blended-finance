"""
Shared design config: event windows + regime definitions.

Built from the FT50 design review. The point is DOUBLE-DISSOCIATION — credit-leg
events (bank exclusion protects, Islamic UP) vs rate-leg events (low-leverage/tech
tilt hurts, Islamic DOWN). Showing both signs on separate clean events identifies a
mechanism rather than a correlation.

All tests run against the MATCHED ex-financials benchmark, not raw S&P, so a
credit-leg "win" cannot be dismissed as "you just don't hold banks".
"""

# name -> (start, end, leg, predicted_sign vs matched benchmark)
EVENTS = {
    "lehman_2008":        ("2008-09-08", "2008-11-21", "credit", "+"),
    "svb_2023":           ("2023-03-08", "2023-03-24", "credit", "+"),
    "credit_suisse_2023": ("2023-03-15", "2023-03-31", "credit", "+"),
    "taper_tantrum_2013": ("2013-05-22", "2013-06-24", "rate",   "-"),  # pure rate, no credit crisis
    "selloff_2018q4":     ("2018-10-01", "2018-12-24", "rate",   "-"),
    "hiking_2022":        ("2022-01-01", "2022-10-31", "rate",   "-"),
    "chatgpt_2022":       ("2022-11-30", "2023-02-28", "ai",     "+"),  # AI-rally onset anchor
    "nvidia_2023":        ("2023-05-24", "2023-07-31", "ai",     "+"),
    "covid_2020":         ("2020-02-19", "2020-03-23", "credit", "+"),
}

# Calendar regimes — DESCRIPTIVE robustness only (not inference).
CALENDAR_REGIMES = {
    "dotcom_2000_02": ("2000-03-01", "2002-10-31"),
    "GFC_2007_09":    ("2007-10-01", "2009-03-31"),
    "covid_2020":     ("2020-02-01", "2020-04-30"),
    "hiking_2022":    ("2022-01-01", "2022-10-31"),
    "ai_2023_now":    ("2023-01-01", None),
}

# State regimes — INFERENCE runs on these (mechanism's own state variables),
# constructed in analysis from macro.csv:
#   credit_stress : hy_oas above/below rolling median  -> credit leg active
#   rate_regime   : sign of fed_funds change over trailing 3m -> hiking/cutting
#   (optional) Markov-switching on returns for endogenous regime detection.
STATE_REGIME_SPEC = {
    "credit_stress": "macro['hy_oas'] > macro['hy_oas'].rolling(252).median()",
    "rate_hiking":   "macro['fed_funds'].diff(63) > 0",
}
