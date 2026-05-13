"""
cf_candidates.py — Collaborative Filtering via Truncated SVD.

This script:
  1. Trains a truncated SVD model on the sparse user-item rating matrix
  2. Evaluates RMSE on a 1% sample of the test set
  3. Generates top-200 CF candidate movies per test user
  4. Saves the model artifacts and candidate dict

Uses scipy.sparse.linalg.svds (no Surprise dependency required).
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
N_FACTORS = 100          # Number of latent factors (rank of SVD)
TOP_K_CANDIDATES = 200   # Candidates per user


# ══════════════════════════════════════════════
#  STEP 1 — Build sparse matrix & train SVD
# ══════════════════════════════════════════════
def train_svd(train: pd.DataFrame):
    """
    Build a user-item sparse rating matrix from training data,
    then compute a rank-k truncated SVD decomposition.

    Returns:
        user_factors : (n_users, k) — U * sqrt(S)
        item_factors : (k, n_items) — sqrt(S) * Vt
        user_means   : per-user mean rating (for bias correction)
        user_to_idx  : dict mapping userId → matrix row index
        item_to_idx  : dict mapping movieId → matrix column index
        idx_to_item  : dict mapping column index → movieId
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Training Truncated SVD model")
    print("=" * 60)

    # Build contiguous index mappings
    unique_users = train["userId"].unique()
    unique_items = train["movieId"].unique()

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {mid: j for j, mid in enumerate(unique_items)}
    idx_to_item = {j: mid for mid, j in item_to_idx.items()}

    n_users = len(unique_users)
    n_items = len(unique_items)
    print(f"\n  Matrix dimensions: {n_users:,} users × {n_items:,} items")

    # Compute per-user mean rating (bias term)
    user_means = train.groupby("userId")["rating"].mean()

    # Build sparse matrix with mean-centred ratings
    rows = train["userId"].map(user_to_idx).values
    cols = train["movieId"].map(item_to_idx).values
    # Centre each rating by subtracting the user's mean
    vals = train["rating"].values - train["userId"].map(user_means).values

    sparse_mat = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    print(f"  Sparse matrix density: {sparse_mat.nnz / (n_users * n_items) * 100:.4f}%")

    # Truncated SVD: A ≈ U · diag(S) · Vt
    print(f"\n  Running SVD with k={N_FACTORS} factors ...")
    t0 = time.time()
    U, S, Vt = svds(sparse_mat.astype(np.float32), k=N_FACTORS)
    elapsed = time.time() - t0
    print(f"  ✓ SVD completed in {elapsed:.1f}s")

    # Absorb singular values into both factors for symmetric scaling
    # user_factors: (n_users, k),  item_factors: (k, n_items)
    sqrt_S = np.sqrt(S)
    user_factors = U * sqrt_S[np.newaxis, :]       # (n_users, k)
    item_factors = np.diag(sqrt_S) @ Vt            # (k, n_items)

    print(f"  User factors shape: {user_factors.shape}")
    print(f"  Item factors shape: {item_factors.shape}")

    return user_factors, item_factors, user_means, user_to_idx, item_to_idx, idx_to_item


# ══════════════════════════════════════════════
#  Predict helper
# ══════════════════════════════════════════════
def predict_rating(user_id, movie_id, user_factors, item_factors,
                   user_means, user_to_idx, item_to_idx):
    """Predict a single (user, movie) rating."""
    if user_id not in user_to_idx or movie_id not in item_to_idx:
        # Fall back to global mean if unknown user/item
        return user_means.mean()
    u_idx = user_to_idx[user_id]
    i_idx = item_to_idx[movie_id]
    pred = user_means.get(user_id, user_means.mean()) + user_factors[u_idx] @ item_factors[:, i_idx]
    # Clip to valid rating range
    return float(np.clip(pred, 0.5, 5.0))


# ══════════════════════════════════════════════
#  STEP 1b — Evaluate RMSE on test sample
# ══════════════════════════════════════════════
def evaluate_rmse(test: pd.DataFrame, user_factors, item_factors,
                  user_means, user_to_idx, item_to_idx):
    """Compute RMSE on a 1% random sample of the test set."""
    print("\n" + "-" * 40)
    print("  Evaluating RMSE on 1% test sample ...")

    sample = test.sample(frac=0.01, random_state=42)
    preds = []
    actuals = []

    for _, row in sample.iterrows():
        pred = predict_rating(
            row["userId"], row["movieId"],
            user_factors, item_factors,
            user_means, user_to_idx, item_to_idx,
        )
        preds.append(pred)
        actuals.append(row["rating"])

    preds = np.array(preds)
    actuals = np.array(actuals)
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))

    print(f"  Test sample size: {len(sample):,}")
    print(f"  ✓ RMSE = {rmse:.4f}")
    return rmse


# ══════════════════════════════════════════════
#  STEP 2 — Generate top-200 CF candidates
# ══════════════════════════════════════════════
def generate_candidates(train: pd.DataFrame, test: pd.DataFrame,
                        user_factors, item_factors,
                        user_means, user_to_idx, item_to_idx, idx_to_item):
    """
    For each user in the test set, predict scores for all movies
    they have NOT rated in the training set, and keep the top-200.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Generating top-200 CF candidates per user")
    print("=" * 60)

    # Precompute each user's set of already-rated movies (from train)
    user_train_items = train.groupby("userId")["movieId"].apply(set).to_dict()

    test_users = test["userId"].unique()
    n_items = item_factors.shape[1]
    all_item_indices = np.arange(n_items)

    candidates = {}
    t0 = time.time()

    for count, uid in enumerate(test_users, 1):
        if uid not in user_to_idx:
            # Unknown user — skip (shouldn't happen with our pipeline)
            continue

        u_idx = user_to_idx[uid]
        u_mean = user_means.get(uid, user_means.mean())

        # Predicted scores for ALL items (vectorised dot product)
        scores = u_mean + user_factors[u_idx] @ item_factors  # shape: (n_items,)
        scores = np.clip(scores, 0.5, 5.0)

        # Mask out already-rated items by setting their score to -inf
        rated_items = user_train_items.get(uid, set())
        rated_indices = [item_to_idx[mid] for mid in rated_items if mid in item_to_idx]
        scores[rated_indices] = -np.inf

        # Get top-200 indices
        top_indices = np.argpartition(scores, -TOP_K_CANDIDATES)[-TOP_K_CANDIDATES:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Store as list of (movieId, predicted_score)
        candidates[uid] = [
            (idx_to_item[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > -np.inf
        ]

        if count % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  ... processed {count:>7,} / {len(test_users):,} users "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\n  ✓ Candidate generation completed in {elapsed:.1f}s")

    return candidates


# ══════════════════════════════════════════════
#  STEP 3 — Save artifacts
# ══════════════════════════════════════════════
def save_artifacts(user_factors, item_factors, user_means,
                   user_to_idx, item_to_idx, idx_to_item,
                   candidates):
    print("\n" + "=" * 60)
    print("STEP 3 — Saving artifacts to data/artifacts/")
    print("=" * 60)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save the SVD model components
    model_path = os.path.join(ARTIFACTS_DIR, "cf_model.pkl")
    model_data = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_means": user_means,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "n_factors": N_FACTORS,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n  ✓ Saved {model_path} ({size_mb:.1f} MB)")

    # Save candidate dict
    cand_path = os.path.join(ARTIFACTS_DIR, "cf_candidates.pkl")
    with open(cand_path, "wb") as f:
        pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(cand_path) / (1024 * 1024)
    print(f"  ✓ Saved {cand_path} ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════
def main():
    print("\n  Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    print(f"  ✓ train:  {train.shape}")
    print(f"  ✓ test:   {test.shape}")
    print(f"  ✓ movies: {movies.shape}")

    # Step 1 — Train SVD
    (user_factors, item_factors, user_means,
     user_to_idx, item_to_idx, idx_to_item) = train_svd(train)

    # Step 1b — Evaluate
    evaluate_rmse(test, user_factors, item_factors,
                  user_means, user_to_idx, item_to_idx)

    # Step 2 — Generate candidates
    candidates = generate_candidates(
        train, test,
        user_factors, item_factors,
        user_means, user_to_idx, item_to_idx, idx_to_item,
    )

    # Step 3 — Save
    save_artifacts(user_factors, item_factors, user_means,
                   user_to_idx, item_to_idx, idx_to_item,
                   candidates)

    # Final summary
    n_users_with_cands = len(candidates)
    avg_cands = np.mean([len(v) for v in candidates.values()]) if candidates else 0
    print("\n" + "=" * 60)
    print("  DONE — Final Summary")
    print("=" * 60)
    print(f"  SVD factors:            {N_FACTORS}")
    print(f"  Users with candidates:  {n_users_with_cands:,}")
    print(f"  Avg candidates/user:    {avg_cands:.1f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
