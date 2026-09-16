# Core V2 market history

Core V2 captures spread and total markets three times per day. The collector requests
`spreads,totals` for the US region, filters out games that have kicked off, and commits
only parsed, validated rows. A raw provider response is retained as a GitHub Actions
artifact for 14 days.

## Durable files

- `data/market/<season>/week_<WW>/market_history.csv`: append-only sportsbook quotes.
- `data/market/<season>/week_<WW>/market_consensus_history.csv`: append-only core-book
  median spreads and totals.
- `data/market/<season>/week_<WW>/book_openers.csv`: immutable first-observed values for
  each game and book.
- `data/market/<season>/week_<WW>/latest_book_lines.csv`: newest quote per game and book.
- `data/tracking/projection_market_snapshots.csv`: the exact Core V2 fair spread joined
  to each market snapshot.
- `data/market/api_usage.csv`: provider usage and remaining-credit audit trail.
- `current_book_lines.csv` and `current_openers.csv`: current Streamlit snapshots.

The workflow stops before making another request when the last recorded remaining-credit
balance is at or below the configured safety threshold (100 by default). Spread and total
capture costs two API credits per scheduled run under the provider's market-by-region
credit model.

The Streamlit refresh button requests both spreads and totals but remains session-only.
Permanent history is written by `.github/workflows/capture_market_history.yml`, which can
also be run manually from GitHub Actions.

This stage captures totals for research and display. It does not yet produce a Core V2
fair-total projection.
