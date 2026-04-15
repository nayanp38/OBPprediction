from __future__ import annotations

import numpy as np
import pandas as pd


def attach_park_factors(batting: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    out = batting.merge(
        park_factors[["Tm", "season", "park_factor"]],
        left_on=["Team", "season"],
        right_on=["Tm", "season"],
        how="left",
    ).drop(columns=["Tm"], errors="ignore")
    out["park_factor"] = out["park_factor"].fillna(1.0)
    return out


def compute_avg_fip_faced(
    pa_level: pd.DataFrame,
    pitcher_fip: pd.DataFrame,
    pitcher_id_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "pitcher_mlbam" in pitcher_fip.columns:
        pa_with_fip = pa_level.merge(
            pitcher_fip[["pitcher_mlbam", "season", "FIP"]],
            left_on=["pitcher", "season"],
            right_on=["pitcher_mlbam", "season"],
            how="left",
        )
    elif pitcher_id_map is not None:
        pa = pa_level.merge(
            pitcher_id_map.rename(
                columns={"key_mlbam": "pitcher_mlbam", "key_fangraphs": "pitcher_fg_id"}
            ),
            left_on="pitcher",
            right_on="pitcher_mlbam",
            how="left",
        )
        pa["pitcher_fg_id"] = pd.to_numeric(pa["pitcher_fg_id"], errors="coerce").astype("Int64")
        pa_with_fip = pa.merge(pitcher_fip, on=["pitcher_fg_id", "season"], how="left")
    else:
        raise ValueError("Either pitcher_mlbam in pitcher_fip or pitcher_id_map must be provided.")

    pa_with_fip = pa_with_fip.dropna(subset=["FIP"]).copy()

    avg_fip = (
        pa_with_fip.groupby(["batter", "season"], as_index=False)["FIP"]
        .mean()
        .rename(columns={"FIP": "avg_fip_faced", "batter": "batter_mlbam"})
    )
    return avg_fip


def attach_avg_fip_to_batting(
    batting: pd.DataFrame,
    avg_fip_faced: pd.DataFrame,
    batter_id_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "key_mlbam" in batting.columns:
        out = batting.merge(
            avg_fip_faced[["batter_mlbam", "season", "avg_fip_faced"]],
            left_on=["key_mlbam", "season"],
            right_on=["batter_mlbam", "season"],
            how="left",
        ).drop(columns=["batter_mlbam"], errors="ignore")
    elif batter_id_map is not None:
        batter_map = batter_id_map[["key_mlbam", "key_fangraphs"]].drop_duplicates().rename(
            columns={"key_mlbam": "batter_mlbam", "key_fangraphs": "IDfg"}
        )
        batter_map["IDfg"] = pd.to_numeric(batter_map["IDfg"], errors="coerce").astype("Int64")
        avg_fip = avg_fip_faced.merge(batter_map, on="batter_mlbam", how="left")
        out = batting.merge(avg_fip[["IDfg", "season", "avg_fip_faced"]], on=["IDfg", "season"], how="left")
    else:
        raise ValueError("Either batting must include key_mlbam or batter_id_map must be provided.")

    out["avg_fip_faced"] = out.groupby("season")["avg_fip_faced"].transform(lambda s: s.fillna(s.mean()))
    return out


def standardize(series: pd.Series) -> pd.Series:
    denom = series.std(ddof=0)
    if pd.isna(denom) or denom == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / denom


def build_feature_matrix(
    batting: pd.DataFrame,
    include_season_dummies: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    out = batting.copy()

    out["K%"] = out["K%"].fillna(out["K%"].median())
    out["GB%"] = out["GB%"].fillna(out["GB%"].median())
    out["Age"] = out["Age"].fillna(out["Age"].median())
    out["avg_fip_faced"] = out["avg_fip_faced"].fillna(out.groupby("season")["avg_fip_faced"].transform("mean"))
    out["avg_fip_faced"] = out["avg_fip_faced"].fillna(out["avg_fip_faced"].mean())

    out["avg_fip_faced_z"] = standardize(out["avg_fip_faced"])
    out["park_factor_z"] = standardize(out["park_factor"])
    out["k_rate_z"] = standardize(out["K%"])
    out["gb_rate_z"] = standardize(out["GB%"])
    out["age_z"] = standardize(out["Age"])

    feature_cols = ["avg_fip_faced_z", "park_factor_z", "k_rate_z", "gb_rate_z", "age_z"]

    if include_season_dummies:
        season_dummies = pd.get_dummies(out["season"], prefix="s", drop_first=True, dtype=float)
        out = pd.concat([out, season_dummies], axis=1)
        feature_cols.extend(season_dummies.columns.tolist())

    out = out.dropna(subset=feature_cols + ["Y", "n"]).reset_index(drop=True)
    return out, feature_cols
