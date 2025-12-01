from datetime import timedelta
import pandas as pd

def compute_weighted_rating(df, rating_col="rating"):
    """
    weighted rating:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
    - R = average rating of item
    - v = number of ratings
    - m = minimum votes
    - C = global mean rating
    """

    C = df[rating_col].mean()
    m = df.groupby("product_id")[rating_col].count()

    stats = df.groupby("product_id")[rating_col].agg(["mean", "count"])
    stats = stats.rename(columns={"mean": "avg_rating", "count": "num_votes"})

    stats = stats[stats["num_votes"] >= m].copy()

    stats["weighted_rating"] = (
        (stats["num_votes"] / (stats["num_votes"] + m)) * stats["avg_rating"]
        + (m / (stats["num_votes"] + m)) * C
    )

    return stats.sort_values("weighted_rating", ascending=False)



def filter_by_time(df, option):
    """Filters dataframe by time range chosen in CLI."""
    if option == "all":
        return df

    max_date = df["datetime"].max()

    if option == "last_month":
        start = max_date - timedelta(days=30)
        return df[df["datetime"] >= start]

    if option == "last_year":
        start = max_date - timedelta(days=365)
        return df[df["datetime"] >= start]

    if option == "custom":
        print("Enter start date (YYYY-MM-DD):")
        s = input("> ")
        print("Enter end date (YYYY-MM-DD):")
        e = input("> ")
        start = pd.to_datetime(s)
        end = pd.to_datetime(e)
        return df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    return df



