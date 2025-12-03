import argparse
import pandas as pd
from datetime import datetime, timedelta
import pickle
import random
from Utils_cleaning import add_datetime_features, clean_rating_dataset,analyze_cutoff_impact
from Utils_app import compute_weighted_rating,filter_by_time,recommend_existing_users


def build_train_matrix(df):
    """Build a user–product matrix."""
    return df.pivot(
        index='user_id',
        columns='product_id',
        values='rating'
    ).fillna(0)

def load_model(model_file: str):
    """Load the saved SVD model."""
    with open(model_file, "rb") as f:
        model_type, model_obj = pickle.load(f)

    if model_type != "svd":
        raise ValueError(f"Loaded model is '{model_type}', but expected 'svd'.")
    return model_obj

def select_users(user_list, train_matrix, n_random):
    """Select users based on the provided list or randomly."""
    available_users = train_matrix.index.tolist()

    if user_list:
        valid = [u for u in user_list if u in available_users]
        if valid:
            return valid
        else:
            print(" None of the provided user IDs exist — selecting random users instead.")

    # If no list provided or none were valid
    n = min(n_random, len(available_users))
    return random.sample(available_users, n)



def main():
    parser = argparse.ArgumentParser(description="Top N best products recommender (weighted rating).")

    parser.add_argument("--file", type=str, default="data/ratings.csv",
                        help="Path to ratings dataset")

    parser.add_argument("--top", type=int, default=20,
                        help="How many top products to show")
    
    parser.add_argument("--model", type=str, default="best_model/best_model.pkl",
                        help="Path to the saved SVD model")

    parser.add_argument("--time", type=str, default="all",
                        choices=["all", "last_month", "last_year", "custom"],
                        help="Time range filter")

    parser.add_argument("--cutoff", type=int, default=7,
                        help="Minimum ratings per product")
    
    parser.add_argument("--n_random_users", type=int, default=3,
                        help="Number of random users if no user list is provided")
    
    parser.add_argument("--user_ids", type=int, nargs="*", default=None,
                        help="List of user IDs, e.g., --user_ids 1 5 18")

    args = parser.parse_args()

    print(" Loading dataset...")
    df = pd.read_csv(args.file)

    print(" Adding datetime features...")
    df = add_datetime_features(df, timestamp_col="timestamp")

    print(" Cleaning dataset...")
    df_clean = clean_rating_dataset(df, timestamp_cols=["timestamp"], rating_col="rating")

    print(" Applying cutoff filtering...")
    df_filtered = analyze_cutoff_impact(df_clean, product_min_ratings=args.cutoff)

    print(f" Applying time filter: {args.time}...")
    df_time = filter_by_time(df_filtered, args.time)

    print(" Computing weighted ratings...")
    ranking = compute_weighted_rating(df_time)

    print("\n=== TOP PRODUCTS ===")
    print(ranking.head(args.top))

    print("\nDone.")


    # --------------------------------------------------------------------
    # RECOMMENDER SYSTEM SECTION (SVD)
    # --------------------------------------------------------------------

    print("\n=== TOP RECOMMENDATIONS ===")

    print("Building train matrix...")
    train_m = build_train_matrix(df_filtered)

    print("Loading model...")
    model = load_model(args.model)

    print("Selecting users...")
    selected_users = select_users(args.user_ids, train_m, args.n_random_users)
    print(f"Selected users: {selected_users}")

    print("Generating recommendations...\n")
    recommendations = recommend_existing_users(
        model,
        train_m,
        top_n=args.top,
        user_ids=selected_users
    )

    print("=== RESULTS ===")
    for uid, recs in recommendations.items():
        print(f"• User {uid} → TOP-{args.top}: {recs}")


if __name__ == "__main__":
    main()