# Utils_eda.py
# Simple EDA functions for ratings dataset: missing values, timestamps, ratings, time analysis
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


def count_unique_values(df):
    """
    Count unique values for user_id, product_id, and rating columns.
    """
    print("Unique values per column:")
    print(f"  user_id: {df['user_id'].nunique()}")
    print(f"  product_id: {df['product_id'].nunique()}")
    print(f"  rating: {df['rating'].nunique()}")

def check_missing_values(df):
    """Count missing values per column"""
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing)
    return missing

def detect_duplicates(df):
    """
    Detects duplicates in the dataset:
    - exact duplicates (all columns identical)
    - duplicate user-product pairs (same user rated same product multiple times)
    
    Returns a dictionary of DataFrames.
    """

    # 1. Exact duplicates (every column identical)
    exact_dupes = df[df.duplicated(keep=False)]

    # 2. User–product duplicate pairs
    # Only rows where both user_id and product_id match more than once
    pair_dupes = (
        df[df.duplicated(subset=["user_id", "product_id"], keep=False)]
        .sort_values(["user_id", "product_id", "timestamp"])
    )

    print("=== DUPLICATE REPORT ===")
    print(f"Exact duplicates: {len(exact_dupes)} rows")
    print(f"Duplicate user-product pairs: {len(pair_dupes)} rows\n")

    return {
        "exact_duplicates": exact_dupes,
        "duplicate_user_product_pairs": pair_dupes
    }

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
    print("ALL values:", df[col].value_counts().to_dict())
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



def plot_top_bottom_users_horizontal(df, top_n=30):
    """
    Plot horizontal bar charts of top and bottom N users by number of ratings.
    X-axis: number of ratings (integers)
    Y-axis: user_id
    """
    counts = df.groupby("user_id")["rating"].count()
    
    # top N users
    top_counts = counts.sort_values(ascending=True).tail(top_n)
    plt.figure(figsize=(10,8))
    ax = top_counts.plot(kind="barh", color='orange')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of ratings")
    plt.ylabel("User ID")
    plt.title(f"Top {top_n} users by number of ratings")
    plt.show()
    
    # bottom N users
    bottom_counts = counts.sort_values(ascending=True).head(top_n)
    plt.figure(figsize=(10,8))
    ax = bottom_counts.plot(kind="barh", color='lightcoral')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of ratings")
    plt.ylabel("User ID")
    plt.title(f"Bottom {top_n} users by number of ratings")
    plt.show()


def plot_top_bottom_products_horizontal(df, top_n=20):
    """
    Plot horizontal bar charts of top and bottom N products by number of ratings.
    X-axis: number of ratings (integers)
    Y-axis: product_id
    """
    counts = df.groupby("product_id")["rating"].count()
    
    # top N products
    top_counts = counts.sort_values(ascending=True).tail(top_n)
    plt.figure(figsize=(10,6))
    ax = top_counts.plot(kind="barh", color='skyblue')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of ratings")
    plt.ylabel("Product ID")
    plt.title(f"Top {top_n} products by number of ratings")
    plt.show()
    
    # bottom N products
    bottom_counts = counts.sort_values(ascending=True).head(top_n)
    plt.figure(figsize=(10,6))
    ax = bottom_counts.plot(kind="barh", color='lightblue')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Number of ratings")
    plt.ylabel("Product ID")
    plt.title(f"Bottom {top_n} products by number of ratings")
    plt.show()

    # Number of products with 1 or two  ratings 
def count_products_by_num_ratings(df):
    """
    Count how many products have 1 rating, 2 ratings, 3 ratings, etc.
    Prints a table and plots a bar chart.
    """
    counts = df.groupby("product_id")["rating"].count()  # liczba ocen na produkt
    freq = counts.value_counts().sort_index()           # ile produktów ma X ocen
    
    print("Number of products by number of ratings:")
    print(freq)
    
def plot_sparsity_matrix(df, max_users=200, max_items=200):
    """
    Visualize sparsity of the user-item rating matrix (binary: rated / not rated).
    Only top active users and top popular items are included to keep the plot readable.
    """
    user_counts = df["user_id"].value_counts().head(max_users).index
    item_counts = df["product_id"].value_counts().head(max_items).index

    subset = df[(df["user_id"].isin(user_counts)) & (df["product_id"].isin(item_counts))]
    pivot = subset.pivot_table(index="user_id", columns="product_id", values="rating", aggfunc="count").fillna(0)

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, aspect='auto', interpolation='nearest')
    plt.title("User-Item Sparsity Matrix (1 = rated, 0 = not rated)")
    plt.xlabel("Product ID")
    plt.ylabel("User ID")
    plt.colorbar(label="Rated")
    plt.show()


def plot_ratings_per_product_hist(df):
    """Plots full histogram of number of ratings per product + limited version (Y<=20)."""
    import matplotlib.pyplot as plt

    counts = df.groupby("product_id")["rating"].count()

    # --- Full histogram ---
    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=30)
    plt.title("Histogram of Number of Ratings per Product")
    plt.xlabel("Number of ratings")
    plt.ylabel("Frequency")
    plt.show()

    # --- Limited Y-axis histogram (max 20) ---
    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=30)
    plt.title("Histogram of Number of Ratings per Product (X-axis limited to 20)")
    plt.xlabel("Number of ratings")
    plt.ylabel("Frequency")
    plt.xlim(0, 20)   # Limit Y-axis
    plt.show()

def plot_ratings_per_product_hist(df):
    """
    Histogram showing how many ratings each product has.
    Helps evaluate item popularity distribution for item-based CF.
    """
    counts = df.groupby("product_id")["rating"].count()

    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=30, edgecolor="black")
    plt.title("Ratings per Product")
    plt.xlabel("Number of ratings")
    plt.ylabel("Number of products")
    plt.show()

def plot_user_similarity_heatmap(df, top_n=100):
    """
    Heatmap of user-user rating correlations for the top N most active users.
    Temporary conversion to categorical to handle sparse IDs.
    """
    # Work on a copy to avoid changing original df
    df_copy = df.copy()
    df_copy["user_id"] = df_copy["user_id"].astype("category")
    df_copy["product_id"] = df_copy["product_id"].astype("category")
    
    top_users = df_copy["user_id"].value_counts().head(top_n).index
    subset = df_copy[df_copy["user_id"].isin(top_users)]
    
    pivot = subset.pivot_table(index="user_id", columns="product_id", values="rating")
    corr = pivot.T.corr(min_periods=2)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title(f"User-User Rating Correlation Heatmap (Top {top_n} Active Users)")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    plt.show()


def plot_item_similarity_heatmap(df, top_n=100):
    """
    Heatmap of item-item rating correlations for the top N most popular products.
    Temporary conversion to categorical to handle sparse IDs.
    """
    df_copy = df.copy()
    df_copy["user_id"] = df_copy["user_id"].astype("category")
    df_copy["product_id"] = df_copy["product_id"].astype("category")
    
    top_products = df_copy["product_id"].value_counts().head(top_n).index
    subset = df_copy[df_copy["product_id"].isin(top_products)]
    
    pivot = subset.pivot_table(index="product_id", columns="user_id", values="rating")
    corr = pivot.T.corr(min_periods=2)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title(f"Item-Item Rating Correlation Heatmap (Top {top_n} Popular Products)")
    plt.xlabel("Product ID")
    plt.ylabel("Product ID")
    plt.show()


def plot_top_products_trends(df, top_n=10):
    """
    Shows rating counts per month for the top N most-rated products.
    Useful for identifying long-term popularity trends.
    """
    df = df.copy()
    df["year_month"] = df["datetime"].dt.to_period("M")

    top_items = df["product_id"].value_counts().head(top_n).index
    subset = df[df["product_id"].isin(top_items)]

    grouped = subset.groupby(["year_month", "product_id"])["rating"].count().unstack(fill_value=0)

    grouped.plot(figsize=(12, 6))
    plt.title(f"Monthly Rating Counts for Top {top_n} Products")
    plt.xlabel("Year-Month")
    plt.ylabel("Number of Ratings")
    plt.grid(True)
    plt.show()

def plot_avg_rating_per_year(df):
    """
    Average rating per year to detect rating drift over time.
    Shows whether ratings become more strict or more lenient.
    """
    df = df.copy()
    df["year"] = df["datetime"].dt.year

    yearly = df.groupby("year")["rating"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly.index, yearly.values, marker="o")
    plt.title("Average Rating per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.ylim(0, 5)
    plt.grid(True)
    plt.show()



def count_sparse_users_products(df, user_threshold=10, product_threshold=5):
    """
    Count users with less than 'user_threshold' ratings
    and products with less than 'product_threshold' ratings.
    """
    # Count ratings per user
    user_counts = df.groupby("user_id")["rating"].count()
    num_users = (user_counts < user_threshold).sum()
    
    # Count ratings per product
    product_counts = df.groupby("product_id")["rating"].count()
    num_products = (product_counts < product_threshold).sum()
    
    print(f"Number of users with less than {user_threshold} ratings: {num_users}")
    print(f"Number of products with less than {product_threshold} ratings: {num_products}")

def count_products_by_num_ratings(df, max_ratings=16):
    """
    Count how many products have exactly X ratings, for X=1..max_ratings.
    Useful to decide filtering thresholds.
    """
    counts = df.groupby("product_id")["rating"].count()
    freq = counts.value_counts().sort_index()

    print(f"Products with 1 to {max_ratings} ratings:")
    for i in range(1, max_ratings+1):
        print(f"  {i} ratings: {freq.get(i, 0)} products")

def analyze_cutoff_impact(df, product_min_ratings=5, user_low_n=10):
    """
    Analyze the impact of filtering products with few ratings and inspect user activity.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['user_id', 'product_id', 'rating'].
        product_min_ratings (int): Minimum number of ratings a product must have to stay.
        user_low_n (int): How many lowest activity levels to display for users.
    
    Returns:
        None
    """
    
    # 1. Product filtering
    count_unique_values(df)

    product_counts = df['product_id'].value_counts()
    print("=== 1. Product rating counts BEFORE filtering ===")
    print(product_counts.describe())
    
    low_rated_products = product_counts[product_counts < product_min_ratings]
    print(f"\nProducts with < {product_min_ratings} ratings: {len(low_rated_products)}")
    
    df_filtered = df[~df['product_id'].isin(low_rated_products.index)]
    print(f"\nRecords BEFORE filtering: {len(df)}")
    print(f"Records AFTER filtering: {len(df_filtered)}")
    print(f"Removed: {len(df) - len(df_filtered)} records ({(len(df) - len(df_filtered))/len(df)*100:.2f}%)")
    count_unique_values(df_filtered)
    # 2. User activity after filtering
    user_activity = df_filtered.groupby('user_id').size().rename('num_ratings')
    total_users = user_activity.shape[0]
    print(f"\n=== 2. User activity (ratings per user) AFTER filtering ===")
    print(f"Total unique users after filtering: {total_users}")
    
    # Lowest N activity levels
    activity_distribution = user_activity.value_counts().sort_index()
    print(f"\n=== Lowest {user_low_n} activity levels (explained) ===")
    print("\nColumns:")
    print("ratings_per_user -> number of ratings a user made")
    print("num_users -> number of users with that number of ratings\n")
    
    lowest_activity_df = activity_distribution.head(user_low_n).reset_index()
    lowest_activity_df.columns = ["ratings_per_user", "num_users"]
    
    

    # # Explanation example for clarity
    for _, row in lowest_activity_df.iterrows():
        print(f"ratings_per_user = {row['ratings_per_user']} → num_users = {row['num_users']}")
    
    return df_filtered