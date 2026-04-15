from __future__ import annotations

import argparse
from pathlib import Path

from obp_pipeline.config import PipelineConfig
from obp_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OBP logistic-normal empirical Bayes pipeline."
    )
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=2023)
    parser.add_argument("--min-qual-pa", type=int, default=50)
    parser.add_argument("--min-model-pa", type=int, default=100)
    parser.add_argument("--include-season-dummies", action="store_true", default=True)
    parser.add_argument("--no-season-dummies", action="store_false", dest="include_season_dummies")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        start_season=args.start_season,
        end_season=args.end_season,
        min_qual_pa=args.min_qual_pa,
        min_model_pa=args.min_model_pa,
        include_season_dummies=args.include_season_dummies,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )
    results = run_pipeline(config)
    cols = [
        "Name",
        "season",
        "n",
        "obp_raw",
        "obp_posterior_mean",
        "obp_ci_lower",
        "obp_ci_upper",
    ]
    print(results[cols].sort_values("obp_posterior_mean", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
