from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/processed") # create data/processed directory
OUT_DIR.mkdir(parents=True, exist_ok=True) # ensure directory exists

SEASON = "2023-24"  # specify the NBA season
SEASON_TYPE = "Regular Season"  # specify the type of season

def pull_team_game_logs(season: str, season_type: str) -> pd.DataFrame:
    """
    Pulls team game logs for a specified NBA season and season type.

    Args:
        season (str): The NBA season in 'YYYY-YY'

        season_type (str): The type of season ('Regular Season', 'Playoffs', etc.)      
    Returns:
        pd.DataFrame: DataFrame containing the team game logs.
    """
    
    # return one row per team per game
    # example: Lakers vs Celtics on 2024-01-01 will have two rows, one for each team
    lg = LeagueGameLog(season=season, season_type_all_star=season_type)
    df = lg.get_data_frames()[0].copy()
    return df

def main():
    df = pull_team_game_logs(SEASON, SEASON_TYPE)
    
    # define the columns we want to keep
    wanted = [
        "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION",
        "MATCHUP", "WL",
        "PTS", "FGA", "FGM", "FG3A", "FG3M",
        "FTA", "FTM",
        "OREB", "DREB", "AST", "TOV", "MIN"
    ]
    
    # select only the wanted columns that are present in the dataframe to prevent code crash
    available = [c for c in wanted if c in df.columns]
    canonical = df[available].copy()

   # Normalize GAME_DATE to datetime format
    if "GAME_DATE" in canonical.columns:
        canonical["GAME_DATE"] = pd.to_datetime(canonical["GAME_DATE"])

    # Save the processed data to CSV
    out_path = OUT_DIR / "team_game_stats.csv"
    canonical.to_csv(out_path, index=False)
    
    # Print summary
    print(f"Saved: {out_path}")
    print(f"Rows: {len(canonical):,}")
    print(canonical.head(3))

if __name__ == "__main__":
    main()