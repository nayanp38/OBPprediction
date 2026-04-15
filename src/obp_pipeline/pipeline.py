from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import (
    enable_pybaseball_cache,
    pull_batting,
    pull_park_factors,
    pull_pitcher_fip,
    pull_statcast_pa_multiple_seasons,
)
from .features import (
    attach_avg_fip_to_batting,
    attach_park_factors,
    build_feature_matrix,
    compute_avg_fip_faced,
)
from .model import extract_obp_posteriors, fit_logistic_normal_eb, summarize_model


def _write_outputs(
    output_dir: Path,
    modeling_dataset: pd.DataFrame,
    posterior_results: pd.DataFrame,
    model_summary: pd.DataFrame,
    beta_summary: pd.DataFrame,
    trace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    modeling_dataset.to_csv(output_dir / "modeling_dataset.csv", index=False)
    posterior_results.to_csv(output_dir / "obp_posterior_estimates.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    beta_summary.to_csv(output_dir / "beta_summary.csv", index=False)
    trace.to_netcdf(output_dir / "obp_trace.nc")


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    enable_pybaseball_cache()
    seasons = config.seasons

    print(f"[info] seasons: {seasons}")
    batting = pull_batting(
        seasons=seasons,
        min_qual_pa=config.min_qual_pa,
        min_model_pa=config.min_model_pa,
    )
    park_factors = pull_park_factors(seasons)
    batting = attach_park_factors(batting, park_factors)

    pa_level = pull_statcast_pa_multiple_seasons(seasons)
    pitcher_fip = pull_pitcher_fip(seasons)

    avg_fip_faced = compute_avg_fip_faced(pa_level, pitcher_fip)

    batting = attach_avg_fip_to_batting(batting, avg_fip_faced)

    modeling_dataset, feature_cols = build_feature_matrix(
        batting,
        include_season_dummies=config.include_season_dummies,
    )
    print(f"[info] final model rows: {len(modeling_dataset)}")
    print(f"[info] feature count: {len(feature_cols)}")

    trace = fit_logistic_normal_eb(
        dataset=modeling_dataset,
        feature_cols=feature_cols,
        draws=config.draws,
        tune=config.tune,
        chains=config.chains,
        target_accept=config.target_accept,
        random_seed=config.random_seed,
    )

    posterior_results = extract_obp_posteriors(modeling_dataset, trace)
    model_summary, beta_summary = summarize_model(trace, feature_cols)
    _write_outputs(
        output_dir=config.output_dir,
        modeling_dataset=modeling_dataset,
        posterior_results=posterior_results,
        model_summary=model_summary,
        beta_summary=beta_summary,
        trace=trace,
    )
    return posterior_results
