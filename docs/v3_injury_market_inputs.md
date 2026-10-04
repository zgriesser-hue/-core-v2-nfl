# V3 injury and market inputs

V3 fair margins update from EPA and reviewed personnel. The betting edge is
shown only when a timestamped market consensus was captured within four hours
of the build. Bookmaker home spreads are negated before comparing them to V3's
home-margin convention. The Streamlit page checks quote age again when viewed;
its manual refresh is session-only.

Official `OUT` QB signals in the injury monitor are review leads. The
`early_availability_<season>.csv` ledger is the reviewed source of QB points.
The continuing-absence register identifies longer injuries requiring a fresh
weekly treatment. If a relevant player lacks a current-week ledger decision,
the game's fair line stays visible and its betting edge is withheld. Review
may record an injury adjustment or `AVAILABLE`, with source and observation
time. No automatic persistence discount is applied; the historical test did
not support one.

The monitor pushes its queue hourly; a queue or market-snapshot commit triggers
the V3 board workflow. A failed workflow leaves the previous published snapshot
in place. Check the board's PVAR snapshot time and market status before using
an edge.

V3 Model Performance reports projection versus close separately from the
market line's CLV. My Bets reports the wager's own CLV where a valid close is
captured. Historical archive-only lines do not become release records.
