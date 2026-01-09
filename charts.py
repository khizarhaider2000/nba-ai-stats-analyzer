import matplotlib.pyplot as plt
import pandas as pd

def plot_leaderboard(df: pd.DataFrame, metric: str, top_n: int = 10, title: str = ""):
    d = df.sort_values(metric, ascending=False).head(top_n)

    plt.figure()
    plt.bar(d["TEAM_ABBREVIATION"], d[metric])
    plt.title(title or f"Top {top_n} Teams by {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_efficiency_landscape(df: pd.DataFrame, x: str = "ORtg", y: str = "DRtg", title: str = ""):
    plt.figure()
    plt.scatter(df[x], df[y])

    # Label points with team abbreviations
    for _, row in df.iterrows():
        plt.text(row[x], row[y], row["TEAM_ABBREVIATION"], fontsize=8)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{x} vs {y}")
    plt.tight_layout()
    plt.show()
