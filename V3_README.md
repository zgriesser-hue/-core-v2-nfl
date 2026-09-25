# v3_model

Clean, tested Python extraction of the frozen/validated V3.0-V3.4
research from your Colab notebook (`V3.ipynb`) into production
modules, fully calibrated on your real historical data and wired
into a live prediction entry point.

## Status

**All-history calibration is done.** Every coefficient in
`config.py` was fit on your real data pulled from Drive on
2026-09-24 (Colab exports of `team_game_efficiency.parquet`,
`v3_6_integrated_pure_v3_oos_predictions.parquet`,
`v3_2_v3_3_combined_oos_predictions.parquet`, and
`v3_3_lagged_market_game_panel.parquet`) - no LOSO fold held out,
no placeholders left. `predict_2026.py` wires rating + spread +
personnel + market + neutral into one function you can call from
your Streamlit app with this week's completed games and next
week's slate.

**Still not included** (separate, later decisions):
- the PVAR personnel/injury system (unfinished/mixed validation in
  the notebook - a bigger call than a calibration pass)
- live weekly data assembly (pulling PBP, injury reports, and
  market lines into the shapes `predict_2026.predict_week` expects)
  and hardening the market-data pipeline
- the Streamlit app itself

## Module map

| Module | Notebook source | What it does |
|---|---|---|
| `rating.py` | V3.0 rebuild cell | Prior-anchored offense/defense rating solver |
| `spread.py` | V3.0 LOSO margin calibration | `home_margin = HFA + beta * strength_diff` |
| `personnel.py` | V3.2d | Major defensive-attrition adjustment |
| `market.py` | V3.3b/V3.3c | Lagged (never-current-game) market-disagreement signal, EWM(5) |
| `neutral.py` | V3.4j | True-neutral-site HFA removal (structural rule) |
| `pipeline.py` | V3.6B | Combines all of the above into `pred_v36` |
| `config.py` | — | Fitted full-history production constants (see docstring for exact provenance of every number) |
| `predict_2026.py` | — | Live entry point: `predict_week(...)` -> `pred_v36` per matchup |

Every module's docstring cites the exact notebook cell it came from,
so you can trace any line back to the original research.

## Validation performed

- `pipeline.predict_v36` reproduces the notebook's own `pred_v36`
  column **exactly** (max error `0.0`) across all 1,218 real games
  in `v3_6_integrated_pure_v3_oos_predictions.csv` - this confirms
  the extraction is faithful, not just structurally similar.
- The offense/defense solver's four tuning constants
  (`OFF_ALPHA`/`OFF_STRENGTH`/`DEF_ALPHA`/`DEF_STRENGTH`) were
  grid-searched on the full 1,218-game sample (2018/2019/2023/2024/
  2025 - 2017 wasn't part of V3.6's frozen sample), refitting HFA/beta
  by OLS at each grid point. The accuracy surface came back nearly
  flat (MAE 10.29-10.35 across the whole grid), consistent with the
  notebook's own finding that Core v1's EPA tuning barely matters
  once other signals are layered on. The best interior point found
  (`OFF_ALPHA=0.6, DEF_ALPHA=0.15, DEF_STRENGTH=16.0`) gives full-
  sample MAE 10.29 / RMSE 13.21 - in the same range as the
  notebook's own Core v1 benchmark (MAE 10.39). A wider or finer grid
  might find a marginally better point, but the gain would be small.
- `predict_2026.predict_week`, run against a real week (2025 week
  10) using this package's production constants, produced `pred_v36`
  within a mean 0.52 points of the notebook's own per-season-LOSO
  numbers for the same games - the expected size of gap, since
  production now uses one full-history alpha instead of that
  season's specific LOSO value.
- `personnel.fit_personnel_adjustment` and `market.fit_market_beta`,
  run full-sample (no fold held out) on the real data, both came out
  with the theoretically-required positive sign, and
  `fit_personnel_adjustment` would have raised instead of silently
  flipping if they hadn't.

## Running the tests

```bash
pip install pytest pandas numpy
cd v3_model
PYTHONPATH=. pytest tests/ -v
```

The suite checks module-level invariants (week-1 returns priors
untouched, `pred_v36` is an exact sum of its components, market
signals never leak the current game, a contradictory personnel-
adjustment sign is rejected) *and* the `predict_2026.predict_week`
wiring itself (matches a manual call to the underlying modules;
neutral-site correctly drops exactly `PRODUCTION_HFA`). It does not
re-run the full-history calibration each time - that was a one-time
research step, not something to redo on every test run.

## Using it live

```python
from v3_model.predict_2026 import predict_week
from v3_model import rating

priors = rating.build_previous_season_priors(team_game_history, season=2026)
predictions = predict_week(team_game_history, priors, matchups)
```

See `predict_2026.py`'s docstring for the exact shape `team_game_history`
and `matchups` need - in particular, `matchups` needs each team's
injury-exposure number and lagged market signal already computed
(this package does the modeling math, not the weekly data assembly).

## Data assembly and Streamlit (done)

`data_assembly.py` turns your ongoing data sources into what
`predict_2026.predict_week` needs:
- `load_team_game_history` - filters your running team_game_efficiency
  file down to games actually played before the target week
- `build_market_signal_table` - builds every team's lagged EWM(5)
  market-disagreement signal from a market-history log
- `build_matchups` - assembles one week's slate (home/away, neutral
  site, market signal) into the shape `predict_week` needs

`board.py` chains all of this into one offline step - run it after
each week's games complete, pointed at next week's slate, and it
writes a CSV:

```bash
python -m v3_model.board \
    --team-game-history team_game_efficiency_running.csv \
    --schedule season_schedule.csv \
    --market-history market_history_log.csv \
    --season 2026 --week 5 \
    --out week_05_v3_board.csv
```

This mirrors your existing Core V2 workflow exactly: the model runs
offline and exports a board CSV; Streamlit only displays it and
compares to a live pull via your existing `parlay_live.py` (`app.py`
never runs model code itself, and neither does `pages/1_V3_Dashboard.py`).

**Deployment: single app, two pages (Option B, decided over two
separate Streamlit Cloud apps).** Put `pages/1_V3_Dashboard.py` in
your `streamlit_export/` folder exactly at that relative path -
Streamlit's built-in multi-page support then adds it as a second
sidebar entry in the SAME app.py deployment automatically, one URL,
no changes to app.py itself. Put a board CSV next to `app.py` (or in
`data/`) as usual; `streamlit run app.py` serves both pages.

Two things that specifically needed handling because this now runs
alongside app.py rather than standalone (see the file's own
docstring for the full explanation):
- Its `ROOT` resolves one level up (`Path(__file__).parent.parent`),
  since it lives in `pages/` but needs the same repo-root paths
  app.py resolves from its own location.
- Every `st.session_state` key it uses is prefixed `v3_`
  (`v3_live_lines` etc.) - session state is shared across all pages
  in a multi-page app, and without the prefix its live-refresh state
  would silently collide with app.py's own same-named keys.

**Important simplification, visible in the app itself**:
`build_matchups` currently sets `home_def_loss`/`away_def_loss` to
0.0 for every game (personnel/PVAR is still shelved per prior
discussion), so this board is currently V3.0 + market only
(`pred_v33`), not the full `pred_v36`. The V3 page shows a banner
saying so. Wire in a real injury-exposure source and remove the 0.0
defaults once PVAR is revisited.

## Reconciliation with the 2026-09-24 development handoff

A separate handoff document (written independently of this package)
identified several real issues here, since fixed:

1. **All `PRODUCTION_*` constants renamed to `EXPERIMENTAL_*`.** They
   were never validated against this project's own bar (V3.7Q), and
   labeling them PRODUCTION was the core mistake the handoff flagged.
   `config.py`'s docstring now gives an honest two-tier provenance
   for every number in the file - see it for details.
2. **V3.7Q's actual result** (pulled from Drive same day): none of
   63 candidate O/D-update formulas reliably beat the frozen
   historical V3.0 control in nested LOSO (weighted MAE 10.399 vs.
   10.391; only 2/6 seasons improved; worse in Weeks 6-10). A
   follow-up confirmation test found freezing the rating at the
   preseason prior all season (no update at all) costs ~0.55 MAE
   relative to V3.0's control - so updating clearly has value, just
   not from any formula tested so far. The live 2026 O/D updater
   remains genuinely unresolved; nothing in this package should be
   read as having settled it.
3. **`predict_2026.py`'s dead-filter bug fixed**: it used to compute
   a leakage-filtered `history` variable and then pass the
   *unfiltered* data into `rating.build_pregame_state` anyway (only
   safe because that function re-filtered internally - luck, not
   defense-in-depth). Now fixed, and `build_pregame_state` itself
   asserts loudly if it ever receives rows at or beyond the target
   week, so this class of bug can't recur silently.
4. **`data_assembly.py` now carries provenance**, not just values:
   `home_market_signal_provenance` distinguishes "observed" from
   "no_prior_games" (genuine zero) from "missing_lookup_failed" (a
   join/data problem that must not be silently treated as a zero).
   `home_personnel_provenance` is `"pvar_not_wired"` for every row -
   explicit placeholder status, not indistinguishable from a real
   zero exposure.

**Not yet reconciled** (needs your actual V3.7Q harness, not mine):
the points<->EPA conversion for 2026 remains unfrozen (see
`preseason_priors_2026.py`'s module docstring) - the priors are
loaded and decomposed correctly, but wiring them into `rating.py`'s
EPA-scale machinery still requires that conversion decision, which
isn't mine to make.

## 2026 preseason priors (done)

`preseason_priors_2026.load_2026_preseason_priors(xlsx_path)` loads
your actual `2026_Deep_Dive_Priors_Core_V2.xlsx` and reproduces the
frozen Andy/Drew decomposition from handoff Section 7 exactly:
`overall_prior` from "Centered Week 1 Prior" (not "Final Week 1
Rating," which the handoff warns bakes in temporary Week-1
adjustments), unit composites from the Offense/Defense sheets with
missing cells filled by column mean, standardized by population SD,
combined into `offense_strength`/`defense_strength` such that they
sum back to `overall_prior` exactly (verified to float precision
against your real uploaded workbook, all 32 teams).

**This output is in points, not EPA** - it does not plug directly
into `rating.build_pregame_state`'s `previous_season_priors` argument,
which expects EPA-like units. Wiring it in requires the points<->EPA
conversion the handoff explicitly says isn't frozen yet; this module
deliberately stops short of inventing that conversion.

## The points<->EPA conversion, resolved by sidestepping it (production_v3.py)

The offense/defense EPA-split machinery in `rating.py` is a separate,
still-open research question (that's literally what V3.7Q tested and
didn't resolve). But **the persistent-margin formula never needed
that split in the first place**: since
`offense_strength + defense_strength == overall_prior` exactly by
construction,

    persistent = (home_off + home_def) - (away_off + away_def)

The split cancels out completely when offense/defense haven't
diverged - but after `epa_updater.update_split_state` moves them
apart based on real in-season performance, this formula reflects the
NEW sum, which is exactly how in-season EPA-based updates move the
projected margin without the split ever needing an EPA<->points
conversion. See `production_v3.py`.

**Revision (2026-09-24, second pass)**: the first version of this
module updated ratings with Core V2's own final-margin-residual rule.
That was flagged and reverted before shipping - it's the exact
mechanism V3 was built to replace (per the handoff's founding
rationale), and reusing it here would have shipped that mistake back
into production. It's now replaced by `epa_updater.py`:

- **In-season updates** use a z-score bridge (details in
  `epa_updater.py`'s docstring): this week's observed EPA is
  standardized against its own cross-league distribution (unit-free),
  blended with the prior's own z-score using asymmetric shrinkage
  (`off_prior_games=8`, `def_prior_games=20`, reused from the V3.7Q
  "Q026" config), then rescaled using the PRIOR's OWN standard
  deviation - never an externally invented conversion factor, and
  works identically whether the prior is in EPA units (historical
  research) or points (the real 2026 Andy/Drew decomposition).
  **Tested honestly, not assumed**: three different EPA-based
  candidates were run through the same nested-LOSO structure V3.7Q
  used, on the real 1,584-game historical sample. None beat V3.0's
  frozen control (10.3909); this one (10.5443) is statistically
  indistinguishable from the best-performing alternative (10.5237,
  0.02 MAE apart - noise) but is the one actually wired in, because
  it's the only one that bridges cleanly to the real 2026 points-scale
  prior without inventing anything. Adding opponent adjustment to
  this candidate was also tried and made it slightly worse (10.5846) -
  reported plainly rather than discarded, since it's informative.
- **V3.2 (personnel) and V3.3 (market signal)** layer on top exactly
  as before - both were always points-scale adjustments
  (`adjustment_points`, a beta applied to `home_margin`), so neither
  one ever depended on the EPA scale.
- **HFA** uses your own `V3Config`'s documented default (`2.20`, per
  handoff Section 20) as-is - not fit by me, not invented, just
  reused. Worth refitting once real 2026 results accumulate.
- **True neutral site** removes HFA exactly as `neutral.py` already
  does everywhere else in this package.

Tested end-to-end against your real Andy/Drew priors and the real
Week 1 2026 schedule (16 games, real market spread lines): predictions
track the market closely (most games within ~1-1.5 points, two larger
disagreements at IND@BAL and LAC@ARI worth a look before trusting
blindly). The update mechanism itself was verified against a
synthetic in-season game: a team with a strong EPA performance moved
up, a team with a poor one moved down, driven entirely by process
metrics - final score/margin is not a parameter anywhere in
`epa_updater.py`'s or `production_v3.py`'s functions (a regression
test enforces this directly, checking no function signature accepts
an actual/observed margin).

**What this does NOT claim to be**: a resolution of V3.7Q, or a
validated improvement over V3.0's control. It's the best-tested
non-margin option available right now, chosen deliberately over
"no update" and over margin-residual (rejected on principle). If a
future test finds something that beats V3.0's control, replace
`epa_updater.update_split_state` with it. That's future
work, not something blocking this path from shipping now.

## Wiring into Core V2's existing automation (staged plan)

Core V2 already has a full automation stack (line capture, closing
capture, model-vs-close grading, a "My Bets" CLV tracker) - see
`core_v2/` and `scripts/` in the main repo. The plan is to reuse as
much of it as possible unchanged, since most of it is genuinely
model-agnostic. Progress so far:

- **Stage 1 (done, no code needed)**: verified `capture_openers.py`,
  `capture_closings.py`, `capture_pregame_closes.py`, and
  `refresh_market_history.py` - and everything they import from
  `core_v2/market.py` and `core_v2/projections.py` - contain zero
  V2-model-specific logic. They write to a model-agnostic path
  (`data/market/{season}/week_{NN}/`), so V2 and V3 share this data
  directly with no duplication.
- **Stage 2 (done - `pages/1_V3_Dashboard.py`)**: live odds refresh wired in,
  reusing `parlay_live.fetch_live_lines` / `rebuild_board` and
  `pregame_filter.select_pregame_game_keys` completely unmodified -
  verified against realistically-shaped synthetic quote data. The
  only V3-specific code is `board_for_rebuild`, a small adapter that
  renames `fair_home_spread_v3` to the generic `fair_home_spread`
  `rebuild_board` expects. Deployed as **Option B**: a second page in
  the same app.py deployment via Streamlit's multi-page `pages/`
  folder, not a separate app - which is why its session-state keys
  are `v3_`-prefixed and its `ROOT` resolves one level up (see the
  file's own docstring). Note: the live refresh button still shares
  the same `PARLAY_API_KEY` secret and API credit pool as Core V2's
  own refresh button.
- **Stage 3 (done - `board.py`)**: `board.py`'s output now includes
  `game_key` in Core V2's exact format
  (`"{season}_W{week}_{away}_{home}"`), alongside V3's own `game_id`
  (not replacing it) - verified byte-for-byte against Core V2's own
  `game_key()` function, with a regression test guarding against
  future drift. This is what Stage 4/5 need: `core_v2.tracking` and
  `core_v2.bet_tracking` both key strictly on this exact string
  format.
- **Stage 4 (done - `scripts/capture_release_v3.py`,
  `scripts/grade_results_v3.py`)**: `capture_release_v3.py` locks
  V3's board + market at release time, writing to separate files
  (`v3_projection_releases.csv`) so V2 and V3 histories never
  collide, but reusing `core_v2.tracking.build_release_records`/
  `append_release_records` UNCHANGED. `grade_results_v3.py` grades
  those releases against close, reusing
  `core_v2.tracking.grade_completed_games` UNCHANGED and the SAME
  shared `closing_consensus.csv` and nflverse results feed V2 uses -
  no separate V3 closing/results capture needed. Both verified
  end-to-end against realistically-shaped ParlayAPI payloads and
  nflverse-shaped results, in `/home/claude/stage4_test` during
  development.

  **A real bug was caught and fixed while testing this**: V3's
  `fair_home_spread_v3` is in margin convention (positive = home
  favored, matching `pred_v30`/`pred_v36`/nflverse's `spread_line`),
  but Core V2's entire tracking/betting stack
  (`market.py`/`rebuild_board`/`build_release_records`) uses standard
  bookmaker convention (negative = home favored) - exact opposites
  for the same fact. This had ALREADY silently inverted
  `board.py`'s own `model_side` (fixed: now `fair > market -> HOME`,
  not `fair < market -> HOME`) and both Stage 2/4's V2-integration
  points, which now explicitly negate (`fair_home_spread =
  -fair_home_spread_v3`) at the boundary, with regression tests
  locking in the correct direction with worked numeric examples.
  `pages/1_V3_Dashboard.py`'s display logic was fixed the same way
  (`fair_display`/`market_display` columns, always bookmaker
  convention, used for every `display_home_line` call). The
  underlying model math (`predict_2026.py`, `production_v3.py`,
  `epa_updater.py`) was never affected - this was purely in the
  display/comparison layer built on top of it.
- **Stage 5 (done - `pages/1_V3_Dashboard.py`, third tab "My Bets")**:
  ports `app.py`'s "My Bets" tab, reusing `core_v2.bet_tracking`
  completely UNCHANGED (bet ledger, `grade_bets_against_closes`, CLV,
  ATS result, P/L via American odds) - confirmed genuinely
  model-agnostic by testing it end-to-end against the real module
  with real Stage 4 grade output. Points at its own bet ledger
  (`data/tracking/my_bets_v3.csv`) and Stage 4's grades file
  (`v3_game_grades.csv`), never V2's `my_bets.csv`/`game_grades.csv`.
  Every entry uses `fair_display`/`market_display` (bookmaker
  convention, from the Stage 4 fix) - never
  `fair_home_spread_v3`/raw `market_home_spread` (margin convention)
  directly, with a regression test locking that in against the real
  `core_v2.bet_tracking` module itself, not a reimplementation.
- **Stage 6 (not started, needs a data-dependency check first)**: an
  EPA-based rating auto-update for V3 using `epa_updater.py` (not
  Core V2's margin-residual `update_ratings.py`, rejected on
  principle) - needs a live weekly team-EPA feed; `build_oa_epa_shadow.py`
  in the main repo may already compute this, worth checking before
  writing anything new.
- **Stage 7 (not started)**: a scheduled workflow (same cadence as
  `publish_projections.yml`) to advance state and rebuild the V3
  board automatically, once Stage 6 exists.

## Suggested next steps

1. **Get `production_v3.py` in front of real 2026 games.** Feed it
   this week's real matchups (schedule + market lines you already
   have) and, once games complete, call `update_overall_rating` with
   the real final margins to advance each team's rating for next
   week. This is the actual shippable path right now.
2. Refit or confirm `EXPERIMENTAL_HFA_DEFAULT` (currently your
   V3Config's own documented 2.20) once you have real 2026 results
   to check it against.
3. Add the `FIXED_PRESEASON` candidate (frozen at preseason prior,
   never updated in-season, otherwise identical evaluation procedure)
   to your real V3.7Q harness as a clean confirmation arm for the
   *separate*, still-open question of whether a split-aware EPA
   updater can ever beat this simple points-scale one - the version
   I ran here was my own reimplementation of the winning family and
   landed further from V3.0's control than your own reported numbers,
   so it's directional evidence only.
4. Get a real `market_history_log.csv` flowing (one row per completed
   game: market line + that game's own `pred_v30` at the time) -
   `test_market_history.csv` in this validation pass used actual
   game outcomes as a stand-in for the market line, which was fine
   for testing the plumbing but is NOT a real market signal; wire in
   your actual `ParlayAPI` opener/closing data here.
5. Decide PVAR's fate, then wire real `home_def_loss`/`away_def_loss`
   into `build_matchups` and drop the 0.0 defaults (and the
   `"pvar_not_wired"` provenance tag along with them).
6. As 2026 games complete, periodically re-run the full-history
   calibration with the new data folded in - none of the
   `EXPERIMENTAL_*` constants are a permanent fact about the model.
