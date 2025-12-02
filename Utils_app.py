from datetime import timedelta
import pandas as pd
import numpy as np


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




def recommend_existing_users(model, train_matrix, top_n=5, user_ids=None):
    """
    Generate top-N recommendations for existing users using a trained SVD model.

    Args:
        model: Trained TruncatedSVD model.
        train_matrix (pd.DataFrame): User x Product rating matrix.
        top_n (int): Number of top recommendations per user.
        user_ids (list or None): List of user_ids to generate recommendations for. 
                                 If None, generate for all users in train_matrix.

    Returns:
        dict: {user_id: list of top-N product_ids}
    """
    # If user_ids not provided, take all users
    if user_ids is None:
        user_ids = train_matrix.index.tolist()
    
    # Transform train_matrix with SVD
    svd_matrix = model.transform(train_matrix)
    approx_matrix = pd.DataFrame(
        np.dot(svd_matrix, model.components_),
        index=train_matrix.index,
        columns=train_matrix.columns
    )
    
    recommendations = {}
    for uid in user_ids:
        if uid not in train_matrix.index:
            continue  # skip users not in training data
        # Sort predicted ratings for user in descending order
        user_preds = approx_matrix.loc[uid]
        # Remove already rated items
        rated_items = train_matrix.loc[uid][train_matrix.loc[uid] > 0].index
        user_preds = user_preds.drop(rated_items)
        # Take top-N
        top_products = user_preds.sort_values(ascending=False).head(top_n).index.tolist()
        recommendations[uid] = top_products

    return recommendations
