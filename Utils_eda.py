# Utils_eda.py
# Simple EDA functions for ratings dataset: missing values, timestamps, ratings, time analysis

import pandas as pd
import matplotlib.pyplot as plt

def add_datetime_column(df, timestamp_col="timestamp", new_col="datetime"):
    """Convert Unix timestamp to readable datetime column"""
    if timestamp_col not in df.columns:
        print(f"Column '{timestamp_col}' not found in dataframe.")
        return df
    df[new_col] = pd.to_datetime(df[timestamp_col], unit="s", errors="coerce")
    print(f"Column '{new_col}' added to dataframe.")
    return df

def show_time_range(df, date_col="datetime"):
    """Show min and max dates"""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    print("Time range:")
    print("Min:", dates.min().strftime("%Y-%m-%d %H:%M:%S"))
    print("Max:", dates.max().strftime("%Y-%m-%d %H:%M:%S"))

def check_missing_values(df):
    """Count missing values per column"""
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing)
    return missing

def show_rows_with_nulls(df):
    """Display rows with at least one NaN"""
    rows_with_nan = df[df.isna().any(axis=1)]
    if rows_with_nan.empty:
        print("No rows with missing values found.")
        return rows_with_nan
    for idx, row in rows_with_nan.iterrows():
        nan_cols = row[row.isna()].index.tolist()
        print(f"Row index {idx} has NaN in columns: {nan_cols}")
    return rows_with_nan

def plot_rating_counts(df):
    """Bar chart of ratings frequency + numerical counts"""
    if "rating" not in df.columns:
        print("Column 'rating' not found in dataframe.")
        return
    counts = df["rating"].value_counts().sort_index()
    print("Number of ratings per value:")
    print(counts)
    plt.figure(figsize=(8,5))
    plt.bar(counts.index.astype(str), counts.values, color='skyblue')
    plt.title("Number of occurrences for each rating")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()

def show_column_stats(df, col):
    """Basic stats for a column"""
    if col not in df.columns:
        print(f"Column '{col}' not found.")
        return
    print(f"Stats for column '{col}':")
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Unique values:", df[col].nunique())
    print("Top 5 values:", df[col].value_counts().head(5).to_dict())
    print()

def plot_ratings_over_time(df, freq="M", date_col="datetime"):
    """Number of ratings over time (daily, weekly, monthly, yearly)"""
    counts = df.set_index(date_col).resample(freq)["rating"].count()
    plt.figure(figsize=(12,5))
    counts.plot()
    plt.title(f"Number of ratings over time (freq='{freq}')")
    plt.xlabel("Time")
    plt.ylabel("Number of ratings")
    plt.show()

def rating_range_and_outliers(df, rating_col="rating", min_valid=0, max_valid=1):
    """
    Show min and max ratings, count how many ratings are at the extremes
    Also prints how many ratings are outside the valid range (optional)
    """
    if rating_col not in df.columns:
        print(f"Column '{rating_col}' not found.")
        return

    min_rating = df[rating_col].min()
    max_rating = df[rating_col].max()
    count_min = (df[rating_col] == min_rating).sum()
    count_max = (df[rating_col] == max_rating).sum()

    print(f"Rating range: min = {min_rating}, max = {max_rating}")
    print(f"Number of ratings at min value: {count_min}")
    print(f"Number of ratings at max value: {count_max}")

    outliers = df[(df[rating_col] < min_valid) | (df[rating_col] > max_valid)]
    print(f"Number of ratings outside valid range [{min_valid}, {max_valid}]: {len(outliers)}")


def count_rows_per_year(df, date_col="datetime"):
    """Count number of rows per year"""
    counts = df.groupby(df[date_col].dt.year).size()
    print("Number of rows per year:")
    print(counts)









