#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import random

# --- Import your cleaning utils ---
from Utils_cleaning import add_datetime_features, clean_rating_dataset,analyze_cutoff_impact


# -------------------------
# Helper functions
# -------------------------
def compute_rmse_mae(pred_matrix, true_matrix):
    mask = true_matrix.values > 0
    mse = mean_squared_error(true_matrix.values[mask], pred_matrix.values[mask])
    mae = mean_absolute_error(true_matrix.values[mask], pred_matrix.values[mask])
    return np.sqrt(mse), mae

def top_n(pred_matrix, user_id, n=5):
    if isinstance(pred_matrix, pd.DataFrame):
        top_items = np.argsort(pred_matrix.loc[user_id].values)[::-1][:n]
        return pred_matrix.columns[top_items].tolist()
    else:
        top_items = np.argsort(pred_matrix[user_id])[::-1][:n]
        return top_items.tolist()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Recommender system script with cleaning/filtering (SVD + Item-Based CF)")
    parser.add_argument("--file", type=str, required=True, help="Path to ratings CSV file")
    parser.add_argument("--n_svd", type=int, default=20, help="Number of components for SVD")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top recommendations per user")
    parser.add_argument("--n_users", type=int, default=3, help="Number of random users to show recommendations for")
    parser.add_argument("--min_ratings", type=int, default=2, help="Minimum ratings per user to include in split")
    parser.add_argument("--cutoff", type=int, default=7, help="Minimum ratings per product after filtering")
    args = parser.parse_args()

    # -------------------------
    # 1️⃣ Load and clean data
    # -------------------------
    print("Loading dataset...")
    df = pd.read_csv(args.file)

    print("Adding datetime features...")
    df = add_datetime_features(df, timestamp_col="timestamp")

    print("Cleaning dataset...")
    df_clean = clean_rating_dataset(df, timestamp_cols=["timestamp"], rating_col="rating")

    print(f"Applying cutoff filtering (min ratings per product = {args.cutoff})...")
    df_filtered = analyze_cutoff_impact(df_clean, product_min_ratings=args.cutoff)

    # -------------------------
    # 2️⃣ User-level train/test split
    # -------------------------
    train_list, test_list = [], []
    unique_users = df_filtered['user_id'].unique()
    for user in unique_users:
        user_data = df_filtered[df_filtered['user_id'] == user]
        if len(user_data) < args.min_ratings:
            train_list.append(user_data)
            continue
        train_u, test_u = train_test_split(user_data, test_size=0.2, random_state=42)
        train_list.append(train_u)
        test_list.append(test_u)
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=df_filtered.columns)

    # -------------------------
    # 3️⃣ Build user x product matrices
    # -------------------------
    train_matrix = train_df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
    test_matrix = test_df.pivot(index='user_id', columns='product_id', values='rating').reindex(
        index=train_matrix.index, columns=train_matrix.columns, fill_value=0
    )

    # -------------------------
    # 4️⃣ Models
    # -------------------------
    # --- SVD ---
    svd = TruncatedSVD(n_components=args.n_svd, random_state=42)
    svd_matrix = svd.fit_transform(train_matrix)
    approx_matrix = np.dot(svd_matrix, svd.components_)

    # --- Item-Based CF ---
    item_matrix = train_matrix.T
    item_similarity = item_matrix.corr(method='pearson').fillna(0)

    common_items = train_matrix.columns.intersection(item_similarity.columns)
    item_similarity = item_similarity.reindex(index=common_items, columns=common_items, fill_value=0)
    ibcf_pred_matrix = train_matrix[common_items].dot(item_similarity)
    ibcf_pred_matrix = ibcf_pred_matrix.reindex(columns=train_matrix.columns, fill_value=0)

    # -------------------------
    # 5️⃣ Evaluation
    # -------------------------
    svd_rmse, svd_mae = compute_rmse_mae(pd.DataFrame(approx_matrix, index=train_matrix.index, columns=train_matrix.columns), test_matrix)
    ibcf_rmse, ibcf_mae = compute_rmse_mae(ibcf_pred_matrix, test_matrix)

    print(f"SVD RMSE: {svd_rmse:.4f}, MAE: {svd_mae:.4f}")
    print(f"Item-Based CF RMSE: {ibcf_rmse:.4f}, MAE: {ibcf_mae:.4f}")

    # -------------------------
    # 6️⃣ TOP-N recommendations
    # -------------------------
    random_users = random.sample(list(train_matrix.index), min(args.n_users, len(train_matrix.index)))
    for user in random_users:
        print(f"\nUser {user} recommendations:")
        print("SVD TOP-{}: {}".format(args.top_n, top_n(pd.DataFrame(approx_matrix, index=train_matrix.index, columns=train_matrix.columns), user, args.top_n)))
        print("Item-Based CF TOP-{}: {}".format(args.top_n, top_n(ibcf_pred_matrix, user, args.top_n)))

    # -------------------------
    # 7️⃣ Save best model
    # -------------------------
    if svd_rmse < ibcf_rmse:
        best_model = ('svd', svd)
    else:
        best_model = ('ibcf', item_similarity)

    os.makedirs('best_model', exist_ok=True)
    model_path = os.path.join('best_model', 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nBest model: {best_model[0]} saved to {model_path}")

if __name__ == "__main__":
    main()
