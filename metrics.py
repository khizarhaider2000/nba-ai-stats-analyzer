# src/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# 1) Row-level derived fields
# -----------------------------
def normalize_team_game_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes types and ensures required columns exist.
    Expects one row per TEAM per GAME.
    """
    df = df.copy()

    # Required columns for v1 team metrics
    required = [
        "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION",
        "PTS", "FGA", "FGM", "FG3A", "FG3M",
        "FTA", "FTM",
        "OREB", "DREB",
        "AST", "TOV",
        "MIN",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def add_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team possessions estimate at the row (team-game) level.
    POSS ≈ FGA + 0.44*FTA - OREB + TOV
    """
    df = df.copy()
    df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
    df["POSS"] = df["POSS"].replace([np.inf, -np.inf], np.nan)
    return df


def add_shooting_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds eFG and TS at the row level.
    """
    df = df.copy()

    # Avoid divide-by-zero with replace(0, np.nan)
    df["eFG"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
    df["TS"] = df["PTS"] / (2 * (df["FGA"] + 0.44 * df["FTA"]).replace(0, np.nan))

    df["eFG"] = df["eFG"].replace([np.inf, -np.inf], np.nan)
    df["TS"] = df["TS"].replace([np.inf, -np.inf], np.nan)
    return df


# -----------------------------
# 2) Time window filtering
# -----------------------------
def apply_time_window_team_games(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    window: "SEASON", "LAST_5", "LAST_10", "LAST_20"
    Returns filtered team-game rows.
    """
    df = df.copy()
    df = df.sort_values(["TEAM_ID", "GAME_DATE"], ascending=[True, False])

    if window == "SEASON":
        return df

    if window.startswith("LAST_"):
        try:
            n = int(window.split("_")[1])
        except Exception as e:
            raise ValueError(f"Invalid window format: {window}") from e

        return df.groupby("TEAM_ID", group_keys=False).head(n)

    raise ValueError(f"Unsupported window: {window}")


# -----------------------------
# 3) Aggregations (team-level)
# -----------------------------
def aggregate_team_offense(df_team_games: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Aggregates team-game rows to team totals for a window.
    Computes: PPG, ORtg, eFG, TS, Pace, AST_RATE, TOV_RATE
    (Defense metrics require opponent pairing; see aggregate_team_with_defense.)
    """
    df = df_team_games.copy()
    df = apply_time_window_team_games(df, window)

    grouped = df.groupby(["TEAM_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        GAMES=("GAME_ID", "count"),
        PTS=("PTS", "sum"),
        FGA=("FGA", "sum"),
        FGM=("FGM", "sum"),
        FG3M=("FG3M", "sum"),
        FTA=("FTA", "sum"),
        OREB=("OREB", "sum"),
        TOV=("TOV", "sum"),
        AST=("AST", "sum"),
        POSS=("POSS", "sum"),
        MIN=("MIN", "sum"),
    )

    # Metrics
    grouped["PPG"] = grouped["PTS"] / grouped["GAMES"].replace(0, np.nan)
    grouped["ORtg"] = 100 * grouped["PTS"] / grouped["POSS"].replace(0, np.nan)
    grouped["eFG"] = (grouped["FGM"] + 0.5 * grouped["FG3M"]) / grouped["FGA"].replace(0, np.nan)
    grouped["TS"] = grouped["PTS"] / (2 * (grouped["FGA"] + 0.44 * grouped["FTA"]).replace(0, np.nan))

    grouped["AST_RATE"] = grouped["AST"] / grouped["POSS"].replace(0, np.nan)
    grouped["TOV_RATE"] = grouped["TOV"] / grouped["POSS"].replace(0, np.nan)

    # Correct team pace: possessions per game
    grouped["PACE"] = grouped["POSS"] / grouped["GAMES"].replace(0, np.nan)

    return grouped


# -----------------------------
# 4) Opponent pairing + defense
# -----------------------------
def pair_team_opponent(df_team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a paired table: one row per TEAM per GAME, with opponent values joined in.

    Output columns include:
    - PTS_FOR, POSS_FOR
    - PTS_AGAINST, POSS_AGAINST
    """
    df = df_team_games.copy()

    base = df[[
        "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION",
        "PTS", "POSS"
    ]].copy()

    base = base.rename(columns={
        "PTS": "PTS_FOR",
        "POSS": "POSS_FOR",
    })

    opp = base.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "TEAM_ABBREVIATION": "OPP_TEAM_ABBREVIATION",
        "PTS_FOR": "PTS_AGAINST",
        "POSS_FOR": "POSS_AGAINST",
    })

    paired = base.merge(
        opp,
        on=["GAME_ID", "GAME_DATE"],
        how="inner"
    )

    # Remove self-row
    paired = paired[paired["TEAM_ID"] != paired["OPP_TEAM_ID"]].copy()

    return paired


def aggregate_team_with_defense(df_team_games: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Computes team ORtg, DRtg, Net Rating from paired opponent data.
    """
    paired = pair_team_opponent(df_team_games)
    paired = paired.sort_values(["TEAM_ID", "GAME_DATE"], ascending=[True, False])

    if window != "SEASON":
        if window.startswith("LAST_"):
            n = int(window.split("_")[1])
            paired = paired.groupby("TEAM_ID", group_keys=False).head(n)
        else:
            raise ValueError(f"Unsupported window: {window}")

    grouped = paired.groupby(["TEAM_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        GAMES=("GAME_ID", "count"),
        PTS_FOR=("PTS_FOR", "sum"),
        PTS_AGAINST=("PTS_AGAINST", "sum"),
        POSS_FOR=("POSS_FOR", "sum"),
        POSS_AGAINST=("POSS_AGAINST", "sum"),
    )

    grouped["ORtg"] = 100 * grouped["PTS_FOR"] / grouped["POSS_FOR"].replace(0, np.nan)
    grouped["DRtg"] = 100 * grouped["PTS_AGAINST"] / grouped["POSS_AGAINST"].replace(0, np.nan)
    grouped["NET_RTG"] = grouped["ORtg"] - grouped["DRtg"]

    grouped["PACE"] = grouped["POSS_FOR"] / grouped["GAMES"].replace(0, np.nan)

    return grouped


# -----------------------------
# 5) One helper to prepare data
# -----------------------------
def prepare_team_games_for_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: normalize -> add possessions -> add shooting.
    Call this once after loading CSV.
    """
    df = normalize_team_game_df(df_raw)
    df = add_possessions(df)
    df = add_shooting_metrics(df)
    return df
