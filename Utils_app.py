from datetime import timedelta
import pandas as pd


def compute_weighted_rating(df, rating_col="rating", min_votes=5):
    """
    Computes weighted rating using IMDB formula:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        R = average rating for the movie
        v = number of votes
        m = minimum votes required to be listed
        C = mean vote across the dataset
    """
    # global mean
    C = df[rating_col].mean()

    # minimum votes threshold
    m = min_votes

    # product stats
    stats = df.groupby("product_id")[rating_col].agg(["mean", "count"])
    stats = stats.rename(columns={"mean": "avg_rating", "count": "num_votes"})

    # keep only items with enough votes
    stats = stats[stats["num_votes"] >= m].copy()

    # weighted rating formula
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



