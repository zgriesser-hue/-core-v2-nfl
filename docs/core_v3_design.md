# Core V3 — Design Contract

Core V3 is a clean walk-forward NFL team-strength model built alongside V2. V2 remains the production/control model until V3 earns promotion out of sample.

## Objective

At every historical or live decision point, estimate each team's current strength using only information that was available at that time, then create a spread independently of the current game's market.

Primary betting objective: identify stale early-week prices and capture closing-line value. Game-margin accuracy remains an important validation metric.

## Projection architecture

Projected margin is composed from distinct, explainable layers:

1. Persistent non-QB team strength
   - preseason prior
   - opponent-adjusted offensive evidence
   - opponent-adjusted defensive evidence
   - reliability/uncertainty weighting as the season progresses

2. Explicit QB value
   - preseason QB prior
   - current-season opponent-adjusted QB evidence
   - sample-size shrinkage
   - starter/backup difference applied explicitly

3. Personnel availability
   - OL
   - skill positions
   - defensive front/EDGE
   - linebacker
   - secondary
   - injuries, returns, suspensions and role changes

4. Context
   - home-field advantage
   - only independently validated situational effects

5. Market information (separate research variant)
   - only market information available before the current week's release decision
   - previous-week closing markets may inform latent team strength
   - current-game closing information can never construct that game's release projection

## Non-goals

V3 will not:
- mechanically reward or punish teams for final score;
- use a generic accumulating performance adjustment;
- add overlapping EPA/QB/personnel signals without controlling for double counting;
- use future games, future market data, postgame starter identity, or full-season normalization in a pregame feature;
- promote a feature because it is intuitively appealing rather than out-of-sample useful.

## Walk-forward contract

For every season and week:

1. Load only information available before that week's projection timestamp.
2. Build/fetch the team's state entering the week.
3. Freeze the pre-market model projection.
4. Compare with the opening/release market separately.
5. After games complete, ingest results and underlying performance for future weeks only.
6. Preserve the frozen historical state so later calculations cannot rewrite it.

All feature engineering, normalization, calibration and parameter selection must obey this chronology. LOSO validation does not repair leakage inside a feature.

## Research sequence

### V3.0 — foundation
- deterministic historical walk-forward harness
- preseason team prior
- separate offense and defense state
- explicit HFA
- frozen weekly state/projection outputs

### V3.1 — QB
- explicit dynamic QB layer
- no QB double counting in persistent team strength

### V3.2 — personnel
- explicit player/unit availability layer
- test incremental value over V3.1

### V3.3 — market-aware variant
- infer team-strength information from prior available market observations
- compare V3 Pure vs V3 Market-Aware

### V3.4 — situational factors
- add only validated effects such as rest/travel/weather when supported

### V3.5 — matchup/style
- test granular interactions after the base architecture is stable

## Evaluation

Every version is compared against the previous V3 version and V2 control.

Required reporting:
- game-margin MAE and RMSE
- performance by held-out season
- Weeks 1-5, 6-10, and 11-18
- release projection versus eventual close
- model-edge buckets at release
- CLV for tracked/qualified decisions when historically available
- calibration stability and coefficient/weight stability

A small aggregate gain with unstable seasons or no early-season benefit is not sufficient for promotion.

## Explainability

Every projection must be decomposable into named components. A weekly change should eventually be attributable to changes such as:

- persistent offense
- persistent defense
- QB
- OL/skill personnel
- defensive personnel
- context/HFA
- market-informed latent-strength update (market-aware variant only)

No unexplained generic performance adjustment is part of V3.

## Production safety

V3 is research/shadow only until explicitly promoted. Development on V3 must not modify V2 production ratings, publishing, tracking, close capture, or Streamlit behavior unless a separate production change is intentionally approved.
