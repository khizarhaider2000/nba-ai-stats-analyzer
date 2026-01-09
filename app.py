# src/app.py
import pandas as pd
from pathlib import Path
from charts import plot_leaderboard, plot_efficiency_landscape

from metrics import (
    prepare_team_games_for_metrics,
    aggregate_team_offense,
    aggregate_team_with_defense,
)

DATA_PATH = Path("data/processed/team_game_stats.csv")


def main():
    # 1) Load canonical data
    df_raw = pd.read_csv(DATA_PATH)

    # 2) Prepare: normalize + possessions + shooting metrics
    df = prepare_team_games_for_metrics(df_raw)

    window = "SEASON"  # change to "SEASON", "LAST_5", "LAST_20" as needed

    # 3) Offense-only metrics (fast sanity checks)
    offense = aggregate_team_offense(df, window)

    top10_offense = offense.sort_values("ORtg", ascending=False).head(10)
    print("\nTop 10 Offenses by ORtg (" + window + ")\n")
    print(top10_offense[["TEAM_ABBREVIATION", "GAMES", "ORtg", "eFG", "TS", "PACE"]])

    # 4) Full ratings using opponent pairing: ORtg, DRtg, Net Rating
    ratings = aggregate_team_with_defense(df, window)
    plot_leaderboard(ratings, metric="NET_RTG", top_n=10, title="Top 10 Teams by Net Rating (Last 10)")
    plot_efficiency_landscape(ratings, x="ORtg", y="DRtg", title="Efficiency Landscape (Last 10)")


    top10_net = ratings.sort_values("NET_RTG", ascending=False).head(10)
    print("\nTop 10 Teams by Net Rating (" + window + ")\n")
    print(top10_net[["TEAM_ABBREVIATION", "GAMES", "ORtg", "DRtg", "NET_RTG", "PACE"]])


if __name__ == "__main__":
    main()
