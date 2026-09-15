# Offense/Defense Rating Split Backtest

## Decision

Keep the current production rating updater unchanged. Run the offense/defense
model as a research shadow until it has more out-of-sample evidence.

The independent split slightly improved the locked 2023-2025 test relative to
the same EPA signal constrained to one combined team rating, but the advantage
was not stable enough to justify production promotion.

## Method

- Source: nflverse weekly team stats and game schedules, 2018-2025.
- Sample: 2,127 played regular-season games.
- Signal: passing EPA plus rushing EPA divided by attempts, sacks, and carries.
- Offense: each team's exponentially weighted offensive EPA per play.
- Defense: the negative of opponents' exponentially weighted offensive EPA per
  play, so better defense is positive.
- Leakage control: features are captured before a game and state updates occur
  only after every game in that week is captured.
- Season transition: prior state is multiplied by the carryover parameter.
- Tuning: 2020-2022 only, using models initially fit on 2018-2019.
- Locked test: coefficients refit on 2018-2022 and evaluated on 2023-2025.
- Hyperparameter grid: EWM alpha, season carryover, and ridge penalty.

## Locked test results

| Model | Validation MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Combined EPA unit rating | 10.2249 | 10.2644 | 13.2636 |
| Independent offense/defense split | 10.1700 | 10.2453 | 13.2338 |
| Closing market | — | 9.7445 | 12.6533 |

The split improved MAE by 0.0191 points and RMSE by 0.0298 points relative to
the combined EPA model. A paired game bootstrap put the 95% interval for the
MAE improvement at -0.0599 to +0.0956, with a 68.6% probability of improvement.
The interval crosses zero.

## Season stability

| Season | Combined MAE | Split MAE | Split improvement |
|---|---:|---:|---:|
| 2023 | 10.4891 | 10.5437 | -0.0546 |
| 2024 | 10.1286 | 9.9570 | +0.1716 |
| 2025 | 10.1754 | 10.2352 | -0.0598 |

The full-period improvement is concentrated in 2024; the split was worse in
2023 and 2025.

## Interpretation

The refit coefficients were 41.55 for offense and 25.29 for defense, equivalent
to an approximate 62%/38% relative weight. This does not support assigning 62%
of a game's score residual to the winning or outperforming team. It supports
maintaining separate pregame offense and defense efficiency states and allowing
the projection regression to weight those states differently.

## Promotion gate

Before production promotion, require both:

1. A positive rolling out-of-sample result through additional 2026 games.
2. Improvement that is not dependent on one season, ideally with the paired
   uncertainty interval above zero.

Run the research script with:

```bash
python scripts/backtest_rating_unit_split.py \
  --output data/validation/rating_unit_split_backtest.csv
```
