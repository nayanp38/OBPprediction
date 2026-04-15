from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def fit_logistic_normal_eb(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int,
) -> az.InferenceData:
    y_obs = dataset["Y"].to_numpy(dtype=int)
    n_obs = dataset["n"].to_numpy(dtype=int)
    x = dataset[feature_cols].to_numpy(dtype=float)
    n_rows = len(dataset)
    n_features = len(feature_cols)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
        tau = pm.HalfNormal("tau", sigma=0.5)
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=n_rows)
        epsilon = pm.Deterministic("epsilon", tau * z)

        mu_fixed = pm.math.dot(x, beta)
        theta = pm.Deterministic("theta", mu_fixed + epsilon)
        p = pm.Deterministic("p", pm.math.sigmoid(theta))
        pm.Binomial("Y", n=n_obs, p=p, observed=y_obs)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=random_seed,
        )
    return trace


def extract_obp_posteriors(
    dataset: pd.DataFrame,
    trace: az.InferenceData,
) -> pd.DataFrame:
    out = dataset.copy()
    n_rows = len(out)
    p_samples = trace.posterior["p"].values.reshape(-1, n_rows)

    out["obp_posterior_mean"] = p_samples.mean(axis=0)
    out["obp_posterior_median"] = np.median(p_samples, axis=0)
    out["obp_posterior_sd"] = p_samples.std(axis=0)
    out["obp_ci_lower"] = np.percentile(p_samples, 2.5, axis=0)
    out["obp_ci_upper"] = np.percentile(p_samples, 97.5, axis=0)
    out["shrinkage"] = 1.0 - (
        (out["obp_posterior_mean"] - out["obp_raw"]).abs()
        / (out["obp_raw"] - out["obp_raw"].mean()).abs().clip(lower=1e-6)
    )
    return out


def summarize_model(trace: az.InferenceData, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_summary = az.summary(trace, var_names=["beta", "tau"], round_to=4).reset_index()
    beta_summary = az.summary(trace, var_names=["beta"], round_to=4).reset_index()
    beta_summary["feature"] = feature_cols
    return model_summary, beta_summary
