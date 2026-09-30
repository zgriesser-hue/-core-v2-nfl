# Early injury projection flow

Week 4 onward, the reviewed ledger in `data/model/early_availability_<season>.csv`
is the fast path between a sourced injury announcement and a published board.

1. Confirm the player's team, week, status and the source publication time. Use
   `CONFIRMED_OUT` only when the player has been ruled out; uncertainty remains a
   scenario, not a confirmed absence. Enter one row for each player and week.
2. For QBs, supply the reviewed negative team-point scenario. These are
   provisional manual values pending a calibrated starter/replacement model.
3. For defensive EDGE, IDL, LB or DB players, enter zero in both
   `adjustment_points` and `out_adjustment_points`. The V3 builder matches
   the exact team and player name to the frozen `v3_player_pvar_<season>.csv`,
   validates the position and GSIS ID, then applies
   `2.25 * pvar_production` for a confirmed absence. A missing/ambiguous
   rating or unsupported offensive position fails the build.
4. A ledger change triggers the V3 board and Core V2 personnel workflows.
   The Core V2 manual personnel layer only consumes QB ledger rows; defensive
   ledger rows do not suppress its ordinary official-report processing.
5. Before complete official report coverage, V3 shows known defensive
   absences only, with `EARLY_PROVISIONAL_KNOWN_DEF_PVAR_ONLY` status.
   Unknown injuries are not treated as known healthy players. Once all
   teams have official coverage, the regular native PVAR calculation applies.
   Any confirmed early defensive loss already in the official audit is
   subtracted before the early supplement is added, preventing double count.
6. Review `week_v3_board.csv` for the specific matchup,
   `def_availability_notes`, `home_def_loss_pvar`,
   `away_def_loss_pvar`, `personnel_status` and the resulting fair line.
   The Streamlit V3 Market Board displays known defensive outs and QB
   availability separately.

The read-only source monitor (`scripts/monitor_injuries.py`) runs from
`.github/workflows/monitor_injuries.yml` hourly during daytime/evening
Eastern. It scans the official NFL news index and the current-week NFL injury
report, then adds exact/unique roster matches to
`data/monitor/injury_review_queue_<season>.csv`. The V3 Early Injury Watch
tab shows source links, signals, first-seen time and potential defensive
points. News headlines and practice designations are leads only: the monitor
never edits the reviewed ledger or changes a projection. Its news index may
miss team-only announcements, so the review process also accepts a direct
official team source. The scanner rejects previous-week dated headlines and
does not reuse a stale prior-week official report.

Player-specific QB valuation and automatic approval of injury news remain
separate work. The ledger is week-specific: carry a multiweek absence into each
future week only after confirming that week's availability.
