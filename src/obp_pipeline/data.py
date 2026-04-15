from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import pybaseball as pb
import requests


def enable_pybaseball_cache() -> None:
    try:
        pb.cache.enable()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] pybaseball cache could not be enabled: {exc}")
    _install_fangraphs_user_agent_patch()


def _install_fangraphs_user_agent_patch() -> None:
    """
    FanGraphs sometimes rejects default python-user-agent requests (HTTP 403).
    pybaseball calls requests.get() without custom headers, so we patch requests.get
    once at runtime to send browser-like headers for FanGraphs URLs only.
    """
    if getattr(requests, "_obp_fg_header_patch_installed", False):
        return

    original_get = requests.get

    def patched_get(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(url, str) and "fangraphs.com" in url.lower():
            headers = dict(kwargs.get("headers") or {})
            headers.setdefault(
                "User-Agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            headers.setdefault("Referer", "https://www.fangraphs.com/")
            kwargs["headers"] = headers
        return original_get(url, *args, **kwargs)

    requests.get = patched_get  # type: ignore[assignment]
    setattr(requests, "_obp_fg_header_patch_installed", True)


def pull_batting(seasons: Iterable[int], min_qual_pa: int, min_model_pa: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    id_map = _load_chadwick_name_id_map()
    for season in seasons:
        season_df = _fetch_batting_player_table_bref(season)
        season_df = _attach_player_ids(season_df, id_map, "Name")
        season_df["season"] = season
        frames.append(season_df)

    batting = pd.concat(frames, ignore_index=True)
    batting["Y"] = batting["H"] + batting["BB"] + batting["HBP"]
    batting["n"] = batting["PA"]
    batting["obp_raw"] = batting["Y"] / batting["n"]

    keep_cols = [
        "Name",
        "IDfg",
        "key_mlbam",
        "season",
        "Team",
        "Y",
        "n",
        "obp_raw",
        "Age",
        "SO",
        "GB%",
        "K%",
    ]
    batting = batting[keep_cols].dropna(subset=["Y", "n", "key_mlbam"])
    batting = batting[batting["n"] >= min_model_pa].copy()
    batting = batting[batting["n"] >= min_qual_pa].copy()
    batting["IDfg"] = batting["IDfg"].astype("Int64")
    batting["key_mlbam"] = batting["key_mlbam"].astype("Int64")
    return batting


def fetch_park_factors(season: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{season}-park-factors.shtml"
    tables = pd.read_html(url)
    if not tables:
        raise ValueError(f"No park factor table found for season {season}")

    pf = tables[0].copy()
    pf.columns = [str(c).strip() for c in pf.columns]
    pf["season"] = season

    if "Basic" in pf.columns:
        base_col = "Basic"
    elif "PF" in pf.columns:
        base_col = "PF"
    else:
        raise KeyError(f"No Basic/PF park factor column for season {season}")

    if "Tm" not in pf.columns:
        raise KeyError(f"No team abbreviation column 'Tm' for season {season}")

    pf["park_factor"] = pd.to_numeric(pf[base_col], errors="coerce") / 100.0
    return pf[["Tm", "season", "park_factor"]]


def pull_park_factors(seasons: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        try:
            frames.append(fetch_park_factors(season))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] park factor fetch failed for {season}: {exc}")
    if not frames:
        print("[warn] could not fetch park factors; defaulting all park factors to 1.0")
        return pd.DataFrame(columns=["Tm", "season", "park_factor"])
    return pd.concat(frames, ignore_index=True)


def pull_statcast_pa_level(year: int) -> pd.DataFrame:
    start_dt = f"{year}-03-20"
    end_dt = f"{year}-11-15"
    print(f"[info] pulling statcast season {year}")
    sc = pb.statcast(start_dt=start_dt, end_dt=end_dt)

    if "game_year" in sc.columns and "season" not in sc.columns:
        sc = sc.rename(columns={"game_year": "season"})
    if "season" not in sc.columns:
        sc["season"] = year

    needed = ["batter", "pitcher", "events", "season"]
    missing = [c for c in needed if c not in sc.columns]
    if missing:
        raise KeyError(f"Statcast result missing columns: {missing}")

    pa = sc[sc["events"].notna()][["batter", "pitcher", "season"]].copy()
    pa["season"] = year
    return pa


def pull_statcast_pa_multiple_seasons(seasons: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        frames.append(pull_statcast_pa_level(season))
    return pd.concat(frames, ignore_index=True)


def pull_pitcher_fip(seasons: Iterable[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    id_map = _load_chadwick_name_id_map()
    for season in seasons:
        season_df = _fetch_pitching_player_table_bref(season)
        season_df = _attach_player_ids(season_df, id_map, "Name")
        season_df["season"] = season
        season_df = season_df[["IDfg", "key_mlbam", "Name", "season", "FIP"]]
        season_df = season_df.rename(columns={"IDfg": "pitcher_fg_id", "key_mlbam": "pitcher_mlbam"})
        frames.append(season_df)
    out = pd.concat(frames, ignore_index=True)
    out["pitcher_fg_id"] = pd.to_numeric(out["pitcher_fg_id"], errors="coerce").astype("Int64")
    out["pitcher_mlbam"] = pd.to_numeric(out["pitcher_mlbam"], errors="coerce").astype("Int64")
    out["FIP"] = pd.to_numeric(out["FIP"], errors="coerce")
    return out.dropna(subset=["pitcher_mlbam", "FIP"])


def batch_reverse_lookup_mlbam_to_fangraphs(ids: list[int], batch_size: int = 200) -> pd.DataFrame:
    ids = [int(v) for v in ids if pd.notna(v)]
    if not ids:
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])

    results: list[pd.DataFrame] = []
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        try:
            lookup = pb.playerid_reverse_lookup(batch, key_type="mlbam")
            results.append(lookup)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] reverse lookup failed for batch {i}: {exc}")
        time.sleep(1)

    if not results:
        return pd.DataFrame(columns=["key_mlbam", "key_fangraphs"])
    return pd.concat(results, ignore_index=True)


def _fetch_batting_player_table_bref(season: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{season}-standard-batting.shtml"
    tables = pd.read_html(url)
    if len(tables) < 2:
        raise ValueError(f"Could not find player batting table for season {season}")

    df = tables[1].copy()
    df = df.rename(columns={"Player": "Name", "Team": "Team", "Age": "Age"})
    df = df[df["Name"].ne("LgAvg per 600 PA")]
    needed = ["Name", "Team", "Age", "PA", "H", "BB", "HBP", "SO"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Batting table missing columns for season {season}: {missing}")

    df["K%"] = pd.to_numeric(df["SO"], errors="coerce") / pd.to_numeric(df["PA"], errors="coerce")
    df["GB%"] = pd.NA
    return df[["Name", "Team", "Age", "PA", "H", "BB", "HBP", "SO", "K%", "GB%"]]


def _fetch_pitching_player_table_bref(season: int) -> pd.DataFrame:
    url = f"https://www.baseball-reference.com/leagues/majors/{season}-standard-pitching.shtml"
    tables = pd.read_html(url)
    if len(tables) < 2:
        raise ValueError(f"Could not find player pitching table for season {season}")

    df = tables[1].copy()
    df = df.rename(columns={"Player": "Name"})
    needed = ["Name", "FIP"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Pitching table missing columns for season {season}: {missing}")
    return df[["Name", "FIP"]]


def _load_chadwick_name_id_map() -> pd.DataFrame:
    reg = pb.chadwick_register()
    reg = reg[["name_first", "name_last", "key_mlbam", "key_fangraphs", "mlb_played_last"]].copy()
    reg["name_first_norm"] = reg["name_first"].astype(str).str.lower().str.strip()
    reg["name_last_norm"] = reg["name_last"].astype(str).str.lower().str.strip()
    reg["mlb_played_last"] = pd.to_numeric(reg["mlb_played_last"], errors="coerce")
    reg = reg.sort_values("mlb_played_last", ascending=False)
    reg = reg.drop_duplicates(subset=["name_first_norm", "name_last_norm"], keep="first")
    return reg


def _attach_player_ids(df: pd.DataFrame, id_map: pd.DataFrame, name_col: str) -> pd.DataFrame:
    out = df.copy()
    split = out[name_col].astype(str).apply(_split_player_name)
    out["name_first_norm"] = split.str[0]
    out["name_last_norm"] = split.str[1]
    out = out.merge(
        id_map[["name_first_norm", "name_last_norm", "key_mlbam", "key_fangraphs"]],
        on=["name_first_norm", "name_last_norm"],
        how="left",
    )
    out["IDfg"] = pd.to_numeric(out["key_fangraphs"], errors="coerce").astype("Int64")
    out["key_mlbam"] = pd.to_numeric(out["key_mlbam"], errors="coerce").astype("Int64")
    out = out.drop(columns=["name_first_norm", "name_last_norm", "key_fangraphs"], errors="ignore")
    return out


def _split_player_name(name: str) -> tuple[str, str]:
    cleaned = (
        str(name)
        .replace("*", "")
        .replace("#", "")
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )
    parts = [p for p in cleaned.split() if p]
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return parts[0].lower(), parts[0].lower()
    return parts[0].lower(), parts[-1].lower()
