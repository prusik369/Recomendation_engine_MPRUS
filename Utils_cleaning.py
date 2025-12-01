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