# Offense/Defense Rating Split Backtest

## Decision

Keep the current production rating updater unchanged. Run the offense/defense
model as a research shadow until it has more out-of-sample evidence.

The independent split slightly improved the locked 2023-2025 test relative to
the same EPA signal constrained to one combined team rating. A second model
updated each unit from its EPA residual against the opposing unit. Neither
advantage was stable enough to justify production promotion.

## Method

- Source: nflverse weekly team stats and game schedules, 2018-2025.
- Sample: 2,127 played regular-season games.
- Signal: passing EPA plus rushing EPA divided by attempts, sacks, and carries.
- Offense: each team's exponentially weighted offensive EPA per play.
- Defense: the negative of opponents' exponentially weighted offensive EPA per
  play, so better defense is positive.
- Opponent-adjusted variant: expected offensive EPA is the offense state minus
  the opposing defense state. The resulting residual updates the offense and
  opposing defense after the entire week is frozen.
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
| Opponent-adjusted offense/defense split | 10.1224 | 10.2522 | 13.2675 |
| Closing market | — | 9.7445 | 12.6533 |

The split improved MAE by 0.0191 points and RMSE by 0.0298 points relative to
the combined EPA model. A paired game bootstrap put the 95% interval for the
MAE improvement at -0.0599 to +0.0956, with a 68.6% probability of improvement.
The interval crosses zero.

Opponent adjustment improved validation MAE but did not beat the raw split on
the locked test. Its test MAE was 0.0069 points worse than the raw split, with a
paired 95% interval of -0.0921 to +0.0772 for its improvement. It beat the raw
split in 2023 and 2024 but lost that gain in 2025.

## Season stability

| Season | Combined MAE | Split MAE | Split improvement |
|---|---:|---:|---:|
| 2023 | 10.4891 | 10.5437 | -0.0546 |
| 2024 | 10.1286 | 9.9570 | +0.1716 |
| 2025 | 10.1754 | 10.2352 | -0.0598 |

The full-period improvement is concentrated in 2024; the split was worse in
2023 and 2025.

### Opponent-adjusted stability

| Season | Raw split MAE | Opponent-adjusted MAE | Opponent-adjusted improvement |
|---|---:|---:|---:|
| 2023 | 10.5437 | 10.4781 | +0.0657 |
| 2024 | 9.9570 | 9.9183 | +0.0388 |
| 2025 | 10.2352 | 10.3603 | -0.1251 |

Opponent adjustment was directionally better in Weeks 1-4 (+0.0622 MAE) and
Weeks 5-9 (+0.0525), but worse in Weeks 10-18 (-0.0712). Because these buckets
were inspected after opening the locked test, that pattern is exploratory and
must not be used to tune another historical version against the same test.

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
