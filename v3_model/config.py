"""
Model constants for the V3 package, with an explicit confidence tier
for every one. NONE of these are "production" in the sense of having
cleared your project's own validation bar (V3.7Q and its successors) -
per the 2026-09-24 development handoff:

    "Any current EXPERIMENTAL_OFF_ALPHA, EXPERIMENTAL_OFF_STRENGTH,
    etc. are provisional until V3.7Q is complete."

V3.7Q has since run (results pulled from Drive same day). Its finding
changes what "provisional" means here: the nested leave-one-season-out
comparison found NONE of its 63 recursive/cumulative update-formula
candidates reliably beat the frozen historical V3.0 control (weighted
nested MAE 10.399 for the best family vs. 10.391 for V3.0 itself; only
2 of 6 seasons improved; worse in Weeks 6-10). A follow-up confirmation
test (FIXED_PRESEASON, no in-season update at all, vs. a from-scratch
reimplementation of the winning family) found freezing the rating at
the preseason prior all season costs ~0.55 MAE relative to V3.0's
control - so in-season updating clearly has real value, but no tested
formula updates better than whatever V3.0's own (lost) historical
generator did. Net effect: the live 2026 O/D updater is UNRESOLVED,
not close to resolved, and not something to freeze from this file.

Tier 1 - EXPERIMENTAL, actively unresolved (V3.7Q found no winner):
    EXPERIMENTAL_OFF_ALPHA / EXPERIMENTAL_OFF_STRENGTH /
    EXPERIMENTAL_DEF_ALPHA / EXPERIMENTAL_DEF_STRENGTH /
    EXPERIMENTAL_HFA / EXPERIMENTAL_BETA
  These came from a grid search + full-sample OLS refit I ran myself
  on 2026-09-24 (accuracy surface nearly flat, MAE 10.29-10.35 across
  the whole grid, on a 1,218-game subsample that excludes 2017). This
  is a DIFFERENT, less rigorous exercise than V3.7Q's proper nested
  nested LOSO shootout on the full 1,584-game sample, and it should
  not be treated as having settled anything V3.7Q didn't already
  settle. Do not wire these into a live prediction and call it
  production. Useful only as a placeholder to keep the package
  runnable for testing.

Tier 2 - EXPERIMENTAL, but fit on the real, authoritative frozen
pred_v30 (not a self-referential refit), and aligned with a decision
the handoff DOES list as frozen (structural choice, not the
coefficient value):
    EXPERIMENTAL_PERSONNEL_THRESHOLD_QUANTILE /
    EXPERIMENTAL_PERSONNEL_THRESHOLD /
    EXPERIMENTAL_PERSONNEL_ADJUSTMENT_POINTS
  Fit via personnel.fit_personnel_adjustment on
  v3_2_v3_3_combined_oos_predictions.csv (1,218 games, no fold held
  out). The frozen decision is "major defensive attrition only,
  direction = away_major_def_loss - home_major_def_loss" (handoff
  Section 12); the coefficient value itself (2.25 here) is my own
  full-sample fit, not the handoff's own reported number (2.23, on a
  1,296-game eligible sample) - close, but from a different sample,
  so still EXPERIMENTAL rather than a confirmed match.

    EXPERIMENTAL_MARKET_SIGNAL_SPAN / EXPERIMENTAL_MARKET_BETA
  Fit via market.fit_market_beta on v3_3_lagged_market_game_panel.csv
  (1,487 eligible games - matches the handoff's own eligible-sample
  count exactly). The frozen decision is "lagged EWM(5), never the
  current game's own line" (handoff Section 13, confirmed selected in
  all six LOSO folds); the beta value (0.64) is my own full-sample
  fit, not independently confirmed against the handoff's own number
  (which isn't given directly, only the resulting MAE improvement).

Per the handoff's own rule: "do not silently replace frozen research
decisions or label newly fitted parameters PRODUCTION." Nothing below
is PRODUCTION_*; everything is EXPERIMENTAL_* until it clears your
project's own validation bar, whatever family that ends up being.
"""

from __future__ import annotations

V3_SEASONS = [2017, 2018, 2019, 2023, 2024, 2025]

# ---------------------------------------------------------------
# V3.0 rating-solver parameters as they appeared in a Colab cell
# ("V3 RESEARCH - REBUILD FROZEN STATES AFTER COLAB RESTART"),
# per-season LOSO-selected values. Per the handoff (Section 5-6),
# the ORIGINAL generator that produced the authoritative frozen
# states was lost, and everything since - including these numbers -
# is reverse-engineering with acknowledged drift. Kept here for
# reference/debugging only; not confirmed to match the authoritative
# v3_0_authoritative_control.parquet exactly.
# ---------------------------------------------------------------

OFF_ALPHA = {
    2017: 0.45,
    2018: 0.40,
    2019: 0.45,
    2023: 0.40,
    2024: 0.50,
    2025: 0.40,
}

OFF_STRENGTH = {season: 6.0 for season in V3_SEASONS}

DEF_ALPHA = {
    2017: 0.20,
    2018: 0.25,
    2019: 0.20,
    2023: 0.20,
    2024: 0.40,
    2025: 0.15,
}

DEF_STRENGTH = {
    2017: 16.0,
    2018: 16.0,
    2019: 20.0,
    2023: 16.0,
    2024: 16.0,
    2025: 24.0,
}

# ---------------------------------------------------------------
# Tier 1 (see module docstring): unresolved, V3.7Q found no winner
# among 63 candidates. These exist only so the package runs
# end-to-end for testing - not a claim that this is the right
# updater or the right conversion/HFA for 2026.
# ---------------------------------------------------------------

EXPERIMENTAL_OFF_ALPHA = 0.6
EXPERIMENTAL_OFF_STRENGTH = 6.0
EXPERIMENTAL_DEF_ALPHA = 0.15
EXPERIMENTAL_DEF_STRENGTH = 16.0
EXPERIMENTAL_HFA = 1.6772070042614708
EXPERIMENTAL_BETA = 54.3922407160589

# ---------------------------------------------------------------
# Tier 2 (see module docstring): structural choice is frozen by the
# handoff; this specific coefficient value is my own fit, on a
# similar but not identical sample to the handoff's own numbers.
# ---------------------------------------------------------------

EXPERIMENTAL_PERSONNEL_THRESHOLD_QUANTILE = 0.90  # frozen structural choice
EXPERIMENTAL_PERSONNEL_THRESHOLD = 0.7502802403321167  # my fit; handoff's own eligible sample differs (1,296 vs 1,218 here)
EXPERIMENTAL_PERSONNEL_ADJUSTMENT_POINTS = 2.252841414784494  # my fit; handoff reports 2.23 on its own sample

EXPERIMENTAL_MARKET_SIGNAL_SPAN = 5  # frozen structural choice (EWM5 selected in all 6 LOSO folds per handoff)
EXPERIMENTAL_MARKET_BETA = 0.6369155503375366  # my fit; not independently confirmed against handoff's own number
