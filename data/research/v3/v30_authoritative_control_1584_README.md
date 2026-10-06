# Historical V3.0 control predictions

`v30_authoritative_control_1584.csv` is a compact export of the supplied
`v3_3_historical_market_audit_panel.parquet`, restricted to identifiers,
actual home margin, and the original `pred_v30` column. It has one row per
game, no missing values, and 1,584 games across 2017–2019 and 2023–2025.
The mean absolute error of `pred_v30` versus `home_margin` is 10.3908885.

`pred_v30` is positive for a projected home win. This export preserves the
predictions from the historical audit panel; it does not recompute them.
The panel does not contain fold assignments, training dates, or the generator
code, so this CSV alone cannot independently prove leave-one-season-out
provenance. It does not contain predictions for the Z_STANDARDIZED_ASYM
updater or 2020–2022 games.

For a fair market comparison, join on `game_id`, validate team/year matches,
then require an observed pregame timestamp for each market quote. Compare
both market and model error on the identical non-null games. Do not use the
legacy `market_home_margin` from the source audit as a signed opening or
closing quote; it has no captured time or stage in that panel.
