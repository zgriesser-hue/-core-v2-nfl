# Core V2 NFL automation

This repository separates the production NFL model from the Colab research notebook. The Streamlit dashboard remains an independent model/market monitor; market lines are not fed into the fair-line calculation.

## What is automated now

- Builds weekly fair spreads from frozen team ratings, home-field advantage, and a validated personnel-adjustment input.
- Reproduces all 16 saved 2026 Week 1 projections to the stored three-decimal precision.
- Captures each game/book opener once and never replaces it with a later line.
- Builds an opener consensus from DraftKings, FanDuel, Bet365, and Caesars; BetMGM remains excluded.
- Publishes season/week-aware `current_*.csv` files used automatically by Streamlit.
- Runs tests and compile checks before publishing projections.
- Pulls the official weekly NFL injury report, matches non-QBs to frozen role priors,
  and applies the validated status weights and personnel coefficient.
- Applies a small, auditable weekly override table for breaking news and QB changes
  that precede the official injury report.
- Keeps the existing 15-minute in-app live-line cache, so repeated clicks in the same window do not consume another API request.
- Locks the model side and market consensus whenever a projection board is published.
- Captures one immutable core-book closing consensus shortly before each kickoff.
- Imports final scores, grades projection error and ATS results, and calculates CLV
  without changing prior game records.

## Important definition

An opener is the first valid quote observed by this automation. It may not be the sportsbook's absolute first public line if the scheduled job begins after that line appeared.

## GitHub setup

1. Upload the repository contents.
2. Add `PARLAY_API_KEY` under **Settings → Secrets and variables → Actions**.
3. Keep the same key in Streamlit Community Cloud under **App settings → Secrets**.
4. Enable GitHub Actions write permission under **Settings → Actions → General → Workflow permissions**.

`Capture next-week openers` runs Mondays at 11:30 UTC and uses one ParlayAPI request. It stages next-week data without changing the live app. `Validate and publish weekly projections` is manually dispatched after that week's validated personnel input is present; only then are the board, openers, and fallback lines promoted together as one dashboard release.

`Refresh weekly personnel` runs Wednesday through Saturday and can also be run
manually. It writes a candidate personnel layer and projection board but does not
promote them to Streamlit. Missing reports and unresolved material roles remain
explicitly provisional. After reviewing the candidate, run `Validate and publish
weekly projections` for that season/week.

`Capture pre-kickoff closing lines` wakes only around normal NFL kickoff windows.
It checks the schedule and existing close history before making a request. A run with
no uncovered game 5–50 minutes from kickoff uses zero ParlayAPI credits. Each game is
closed once; BetMGM is excluded and at least two core books are required.

`Grade completed games` runs Monday through Wednesday. It imports final scores and
appends one permanent grade per game after both a release record and closing record
exist. The Streamlit Model Performance tab then shows live-season MAE, the ATS record,
and average closing-line value. Historical validation remains displayed separately.

Every manual `Validate and publish weekly projections` run uses one ParlayAPI request
to lock the current market alongside that exact model release. Releases after the
recorded close are never used as the CLV entry reference.

## Local commands

```bash
python -m pip install -e ".[test]"
python -m pytest -q

python -m scripts.publish_week \
  --season 2026 --week 1 \
  --schedule data/schedules/schedule_2026.csv \
  --ratings data/model/team_ratings_2026.csv \
  --personnel data/production/2026/week_01/personnel_adjustments.csv \
  --locks data/model/historical_projection_locks.csv \
  --output current_live_board.csv
```

## Weekly breaking-news adjustments

Edit `data/model/weekly_team_adjustments_2026.csv` in GitHub when confirmed news
must be reflected before the official report. Negative points weaken a team; positive
points strengthen it. Set `exclude_from_auto=True` when the named non-QB would
otherwise also appear in the official injury calculation.

## Migration boundary

The projection assembler, market/opener path, and frozen weekly personnel calculation
are productionized. Personnel refreshes stage a candidate instead of automatically
promoting it, preserving a human review gate for breaking news, QB changes, fuzzy
player matches, and official-report coverage.

Research and model-improvement experiments should remain outside the production path until their out-of-sample validation is approved.
