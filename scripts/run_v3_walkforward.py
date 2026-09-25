from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core_v2.v3 import V3Config, no_update_state, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Core V3.0 walk-forward foundation harness.")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/research/v3"))
    parser.add_argument("--hfa", type=float, default=2.20)
    args = parser.parse_args()

    schedule = pd.read_csv(args.schedule)
    priors = pd.read_csv(args.priors)
    projections, states = run_walk_forward(
        schedule=schedule,
        preseason_priors=priors,
        update_state=no_update_state,
        config=V3Config(default_hfa=args.hfa),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projection_path = args.out_dir / "v3_0_frozen_projections.csv"
    state_path = args.out_dir / "v3_0_weekly_states.csv"
    projections.to_csv(projection_path, index=False)
    states.to_csv(state_path, index=False)

    print("=" * 78)
    print("CORE V3.0 WALK-FORWARD FOUNDATION")
    print("=" * 78)
    print(f"Projection rows: {len(projections):,}")
    print(f"State rows:      {len(states):,}")
    if not projections.empty:
        print(f"Seasons:         {sorted(projections['season'].unique().tolist())}")
        print(f"Weeks:           {int(projections['week'].min())}-{int(projections['week'].max())}")
        print(f"Duplicate games: {int(projections[['season','week','game_id']].duplicated().sum())}")
        print(f"Missing margins: {int(projections['projected_home_margin'].isna().sum())}")
    print(f"Saved projections: {projection_path}")
    print(f"Saved states:      {state_path}")


if __name__ == "__main__":
    main()
