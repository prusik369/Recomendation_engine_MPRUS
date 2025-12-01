import argparse
import pandas as pd
from datetime import datetime, timedelta

from Utils_cleaning import add_datetime_features, clean_rating_dataset
from Utils_eda import analyze_cutoff_impact
from Utils_app import compute_weighted_rating,filter_by_time

def main():
    parser = argparse.ArgumentParser(description="Top N best products recommender (weighted rating).")

    parser.add_argument("--file", type=str, default="data/ratings.csv",
                        help="Path to ratings dataset")

    parser.add_argument("--top", type=int, default=20,
                        help="How many top products to show")

    parser.add_argument("--time", type=str, default="all",
                        choices=["all", "last_month", "last_year", "custom"],
                        help="Time range filter")

    parser.add_argument("--cutoff", type=int, default=7,
                        help="Minimum ratings per product")

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


if __name__ == "__main__":
    main()