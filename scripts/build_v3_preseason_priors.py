from __future__ import annotations

import argparse
from pathlib import Path

from core_v2.v3_priors import DeepDivePriorConfig, load_deep_dive_priors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Core V3 Andy/Drew preseason priors.")
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/research/v3/v3_preseason_priors.csv"))
    parser.add_argument("--composition-scale", type=float, default=1.0)
    args = parser.parse_args()

    priors = load_deep_dive_priors(
        args.workbook,
        args.season,
        DeepDivePriorConfig(composition_scale_points=args.composition_scale),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(args.out, index=False)

    print("=" * 78)
    print("CORE V3 PRESEASON PRIOR BUILD")
    print("=" * 78)
    print(f"Teams: {len(priors)}")
    print(f"Whole-team prior mean: {priors['overall_prior'].mean():.6f}")
    print(f"Reconciliation max error: {(priors['reconciled_total'] - priors['overall_prior']).abs().max():.12f}")
    print(f"Missing unit cells: {int(priors['unit_missing_count'].sum())}")
    print(f"Composition scale: {args.composition_scale:.3f} points per offense-defense z gap")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
