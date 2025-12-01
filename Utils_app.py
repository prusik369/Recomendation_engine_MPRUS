import pandas as pd

def add_datetime_features(df, timestamp_col="timestamp"):
    """
    Converts a timestamp column to datetime and adds new integer columns:
      - year
      - month
      - day
    Parameters:
      df: pd.DataFrame
      timestamp_col: name of the timestamp column (int, float, or str)
    Returns:
      df: pd.DataFrame with new columns ['datetime', 'year', 'month', 'day']
    """
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df["datetime"] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce')
    else:
        df["datetime"] = df[timestamp_col]

    # Extract year, month, day as integers
    df["year"] = df["datetime"].dt.year.astype('Int64')
    df["month"] = df["datetime"].dt.month.astype('Int64')
    df["day"] = df["datetime"].dt.day.astype('Int64')
    
    return df

def remove_nulls_except_timestamp(df, timestamp_cols=["timestamp"]):
    """
    Remove all rows with null values except for timestamp/datetime columns.
    Prints the number of removed rows.
    """
    before = len(df)
    cols_to_check = [col for col in df.columns if col not in timestamp_cols]
    df_clean = df.dropna(subset=cols_to_check)
    after = len(df_clean)
    print(f"Removed {before - after} rows containing nulls (excluding timestamp columns).")
    return df_clean

def remove_user_product_duplicates(df, user_col="user_id", product_col="product_id"):
    """
    Remove duplicates where the same user rated the same product multiple times.
    Keeps the first occurrence and prints the number of removed rows.
    """
    before = len(df)
    df_clean = df.drop_duplicates(subset=[user_col, product_col], keep='first')
    after = len(df_clean)
    print(f"Removed {before - after} duplicate user-product ratings.")
    return df_clean

def remove_min_max_ratings(df, rating_col="rating"):
    """
    Remove rows with minimum or maximum ratings in the dataset.
    Prints the number of removed rows.
    """
    min_rating = df[rating_col].min()
    max_rating = df[rating_col].max()
    before = len(df)
    df_clean = df[(df[rating_col] != min_rating) & (df[rating_col] != max_rating)]
    after = len(df_clean)
    print(f"Removed {before - after} rows with min ({min_rating}) or max ({max_rating}) ratings.")
    return df_clean

def dataset_statistics(df, user_col="user_id", product_col="product_id"):
    """
    Prints basic statistics of the dataset:
      - Number of rows
      - Number of unique users
      - Number of unique products
      - Average number of ratings per user
      - Average number of ratings per product
    """
    num_rows = len(df)
    num_users = df[user_col].nunique()
    num_products = df[product_col].nunique()
    avg_ratings_per_user = df.groupby(user_col).size().mean()
    avg_ratings_per_product = df.groupby(product_col).size().mean()
    
    print("Dataset statistics:")
    print(f"  Rows: {num_rows}")
    print(f"  Unique users: {num_users}")
    print(f"  Unique products: {num_products}")
    print(f"  Avg ratings per user: {avg_ratings_per_user:.2f}")
    print(f"  Avg ratings per product: {avg_ratings_per_product:.2f}")

def clean_rating_dataset(df, timestamp_cols=["timestamp"], rating_col="rating", user_col="user_id", product_col="product_id"):
    """
    Full cleaning pipeline:
      1. Print initial dataset statistics
      2. Remove nulls (except timestamp)
      3. Remove user-product duplicates
      4. Remove min/max ratings
      5. Print final dataset statistics
    Returns the cleaned DataFrame.
    """
    print("=== Initial dataset statistics ===")
    dataset_statistics(df, user_col=user_col, product_col=product_col)
    
    df_clean = remove_nulls_except_timestamp(df, timestamp_cols=timestamp_cols)
    df_clean = remove_user_product_duplicates(df_clean, user_col=user_col, product_col=product_col)
    df_clean = remove_min_max_ratings(df_clean, rating_col=rating_col)
    
    print("=== Final cleaned dataset statistics ===")
    dataset_statistics(df_clean, user_col=user_col, product_col=product_col)
    
    return df_clean

import pandas as pd

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
    product_counts = df['product_id'].value_counts()
    print("=== 1. Product rating counts BEFORE filtering ===")
    print(product_counts.describe())
    
    low_rated_products = product_counts[product_counts < product_min_ratings]
    print(f"\nProducts with < {product_min_ratings} ratings: {len(low_rated_products)}")
    
    df_filtered = df[~df['product_id'].isin(low_rated_products.index)]
    print(f"\nRecords BEFORE filtering: {len(df)}")
    print(f"Records AFTER filtering: {len(df_filtered)}")
    print(f"Removed: {len(df) - len(df_filtered)} records ({(len(df) - len(df_filtered))/len(df)*100:.2f}%)")
    
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
    
    # Explanation example for clarity
    for _, row in lowest_activity_df.iterrows():
        print(f"ratings_per_user = {row['ratings_per_user']} → num_users = {row['num_users']}")
    
    return df_filtered
