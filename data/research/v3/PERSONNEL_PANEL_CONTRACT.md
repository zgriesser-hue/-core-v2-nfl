# V3 historical personnel panel contract

One row per historical NFL game, with every feature restricted to information available before kickoff.

Required fields: season, home_margin, pred_v30, home_def_loss_pvar, away_def_loss_pvar, home_def_max_player_loss, away_def_max_player_loss, and home/away def_edge_loss, def_idl_loss, def_lb_loss, def_db_loss. Week, game_id, home_team and away_team are recommended audit fields.

Personnel fields must be reconstructed from injury/status information knowable pregame. Final scores and later availability are labels/audit information only. pred_v30 must be the frozen historical V3.0 pregame prediction, never a refit using the held-out result.

scripts/test_v3_personnel_specs.py uses no-intercept fits and leave-one-season-out folds when multiple seasons exist. Production is never changed by the runner. The historical V3.2 binary-major-loss rule remains the control; do not promote a richer specification from aggregate MAE alone. Require cross-season stability and sensible coefficient signs and magnitudes.
