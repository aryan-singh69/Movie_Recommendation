"""
content_candidates.py — Content-based candidate generation using TF-IDF + FAISS.

This script:
  1. Builds a TF-IDF matrix from the movie `tags` column
  2. Builds a FAISS IndexFlatIP index for fast cosine-similarity search
  3. Generates top-100 content candidates per test user based on their
     rating-weighted content profile
  4. Saves all artifacts (vectorizer, matrix, index, candidates, mappings)

No ML modeling or ranking — only content-based retrieval.
"""

import gc
import os
import pickle
import time
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts")

MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
MAX_FEATURES = 10_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
TOP_K_CANDIDATES = 100
# Retrieve extra items from FAISS so we still have ≥100 after removing rated ones
FAISS_QUERY_K = 200


# ══════════════════════════════════════════════
#  STEP 1 — Build TF-IDF matrix
# ══════════════════════════════════════════════
def build_tfidf(movies: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the `tags` column and return
    the sparse TF-IDF matrix and the fitted vectorizer.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Building TF-IDF matrix")
    print("=" * 60)

    # Fill any missing tags with empty string
    tags = movies["tags"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english",
    )

    tfidf_matrix = vectorizer.fit_transform(tags)
    print(f"\n  ✓ TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Non-zero entries: {tfidf_matrix.nnz:,}")

    return vectorizer, tfidf_matrix


# ══════════════════════════════════════════════
#  STEP 2 — Build FAISS index
# ══════════════════════════════════════════════
def build_faiss_index(tfidf_matrix):
    """
    Convert the sparse TF-IDF matrix to dense float32, L2-normalise,
    and add to a FAISS IndexFlatIP (inner product = cosine similarity
    after normalisation).
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Building FAISS index")
    print("=" * 60)

    # Convert sparse → dense float32
    if issparse(tfidf_matrix):
        dense = tfidf_matrix.toarray().astype(np.float32)
    else:
        dense = np.asarray(tfidf_matrix, dtype=np.float32)

    # L2-normalise so dot product = cosine similarity
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for empty tag vectors
    dense_normed = dense / norms

    # Build flat inner-product index
    dim = dense_normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_normed)

    print(f"\n  ✓ FAISS index built")
    print(f"  Dimension: {dim}")
    print(f"  Total vectors: {index.ntotal:,}")

    return index, dense_normed


# ══════════════════════════════════════════════
#  STEP 3 — Generate top-100 content candidates
# ══════════════════════════════════════════════
def generate_candidates(train: pd.DataFrame, test: pd.DataFrame,
                        dense_normed: np.ndarray, index: faiss.IndexFlatIP,
                        movie_id_to_idx: dict, idx_to_movie_id: dict):
    """
    For each test user:
      1. Build a content profile = rating-weighted average of TF-IDF vectors
         (vectorised via sparse user-item matrix × dense TF-IDF matrix)
      2. Batch-query FAISS for nearest movies
      3. Exclude already-rated movies, keep top-100
    """
    from scipy.sparse import csr_matrix as csr

    print("\n" + "=" * 60)
    print("STEP 3 — Generating top-100 content candidates per user")
    print("=" * 60)

    # ── Identify test users & build contiguous user index ──
    test_user_set = set(test["userId"].unique())
    # Filter train to only test users to save memory
    train_test = train[train["userId"].isin(test_user_set)]

    test_users_sorted = sorted(test_user_set)
    user_to_uidx = {uid: i for i, uid in enumerate(test_users_sorted)}
    n_users = len(test_users_sorted)
    n_movies = len(movie_id_to_idx)
    print(f"\n  Test users: {n_users:,},  Movies: {n_movies:,}")

    # ── Build sparse user-item rating matrix ──
    print("  Building sparse user-item matrix ...")
    t0 = time.time()

    uids_arr = train_test["userId"].values
    mids_arr = train_test["movieId"].values
    rats_arr = train_test["rating"].values.astype(np.float32)

    # Filter to only entries where the movieId exists in our index
    mask = np.array([m in movie_id_to_idx for m in mids_arr])
    rows = np.array([user_to_uidx[u] for u in uids_arr[mask]], dtype=np.int32)
    cols = np.array([movie_id_to_idx[m] for m in mids_arr[mask]], dtype=np.int32)
    rats_filtered = rats_arr[mask]

    ui_sparse = csr((rats_filtered, (rows, cols)), shape=(n_users, n_movies))
    del rows, cols, rats_filtered, rats_arr, uids_arr, mids_arr, mask

    # Normalise each user's row so weights sum to 1 (weighted average)
    row_sums = np.array(ui_sparse.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    from scipy.sparse import diags
    norm_diag = diags(1.0 / row_sums)
    ui_normed = norm_diag @ ui_sparse
    del ui_sparse, norm_diag, row_sums

    print(f"  ✓ Sparse matrix built in {time.time() - t0:.1f}s")

    # ── Precompute per-user rated movie sets ──
    print("  Building per-user rated sets ...")
    user_rated = defaultdict(set)
    for u, m in zip(train_test["userId"].values, train_test["movieId"].values):
        user_rated[u].add(m)

    # Free the train subset — no longer needed, reclaims ~1 GB
    del train_test
    gc.collect()

    # ── Batch: compute profiles → FAISS query → save candidates ──
    # Save candidates incrementally to disk to avoid accumulating in memory
    BATCH = 500
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    cand_path = os.path.join(ARTIFACTS_DIR, "content_candidates.pkl")
    candidates_count = 0
    total_cands = 0
    all_candidates = {}
    t1 = time.time()

    for batch_start in range(0, n_users, BATCH):
        batch_end = min(batch_start + BATCH, n_users)

        # Compute profiles for this batch: (batch_size, dim)
        batch_ui = ui_normed[batch_start:batch_end]
        batch_profiles = (batch_ui @ dense_normed).astype(np.float32)

        # L2-normalise for cosine similarity
        norms = np.linalg.norm(batch_profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batch_profiles /= norms

        # Query FAISS
        scores, indices = index.search(batch_profiles, FAISS_QUERY_K)

        # Collect candidates, filtering out already-rated movies
        for local_i in range(batch_end - batch_start):
            global_i = batch_start + local_i
            uid = test_users_sorted[global_i]
            rated_set = user_rated.get(uid, set())

            user_cands = []
            for j in range(FAISS_QUERY_K):
                idx = int(indices[local_i, j])
                if idx < 0:
                    continue
                mid = idx_to_movie_id.get(idx)
                if mid is None or mid in rated_set:
                    continue
                user_cands.append((mid, float(scores[local_i, j])))
                if len(user_cands) >= TOP_K_CANDIDATES:
                    break

            if user_cands:
                all_candidates[uid] = user_cands
                candidates_count += 1
                total_cands += len(user_cands)

        del batch_profiles, scores, indices
        gc.collect()

        # Print progress every ~10K users
        if batch_end % 10_000 < BATCH:
            elapsed = time.time() - t1
            print(f"  ... processed {batch_end:>7,} / {n_users:,} users "
                  f"({elapsed:.1f}s elapsed, {candidates_count:,} with candidates)")

    elapsed = time.time() - t1
    print(f"\n  ✓ Candidate generation completed in {elapsed:.1f}s")
    skipped = n_users - candidates_count
    avg_cands = total_cands / max(candidates_count, 1)
    print(f"  Users with candidates: {candidates_count:,}")
    print(f"  Avg candidates/user:   {avg_cands:.1f}")
    print(f"  Skipped users:         {skipped}")

    return all_candidates


# ══════════════════════════════════════════════
#  STEP 4 — Save artifacts
# ══════════════════════════════════════════════
def save_artifacts(vectorizer, tfidf_matrix, index,
                   candidates, movie_id_to_idx, idx_to_movie_id):
    print("\n" + "=" * 60)
    print("STEP 4 — Saving artifacts to data/artifacts/")
    print("=" * 60)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    def _save_pkl(obj, name):
        path = os.path.join(ARTIFACTS_DIR, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ Saved {name} ({size_mb:.1f} MB)")

    _save_pkl(vectorizer, "tfidf_vectorizer.pkl")
    _save_pkl(tfidf_matrix, "tfidf_matrix.pkl")

    # NOTE: Skipping faiss_index.bin (375 MB) to save disk space.
    # The index can be rebuilt from tfidf_matrix.pkl in ~1 second:
    #   dense = tfidf_matrix.toarray().astype(np.float32)
    #   dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    #   index = faiss.IndexFlatIP(dense.shape[1]); index.add(dense)
    print("  ⓘ Skipped faiss_index.bin (can be rebuilt from tfidf_matrix.pkl)")

    _save_pkl(candidates, "content_candidates.pkl")
    _save_pkl(movie_id_to_idx, "movie_id_to_idx.pkl")
    _save_pkl(idx_to_movie_id, "idx_to_movie_id.pkl")


# ══════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════
def main():
    print("\n  Loading data ...")
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"  ✓ movies: {movies.shape}")
    print(f"  ✓ train:  {train.shape}")
    print(f"  ✓ test:   {test.shape}")

    # Build movieId ↔ index mappings (row position in movies_final.csv = FAISS index)
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies["movieId"])}
    idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

    # Step 1 — TF-IDF
    vectorizer, tfidf_matrix = build_tfidf(movies)

    # Step 2 — FAISS index
    index, dense_normed = build_faiss_index(tfidf_matrix)

    # Step 3 — Generate candidates
    candidates = generate_candidates(
        train, test,
        dense_normed, index,
        movie_id_to_idx, idx_to_movie_id,
    )
    # Free large objects no longer needed
    del train, test, dense_normed
    gc.collect()

    # Step 4 — Save
    save_artifacts(vectorizer, tfidf_matrix, index,
                   candidates, movie_id_to_idx, idx_to_movie_id)

    # Final summary
    n_users = len(candidates)
    avg_cands = np.mean([len(v) for v in candidates.values()]) if candidates else 0
    print("\n" + "=" * 60)
    print("  DONE — Final Summary")
    print("=" * 60)
    print(f"  TF-IDF features:           {MAX_FEATURES}")
    print(f"  FAISS index vectors:       {index.ntotal:,}")
    print(f"  Users with candidates:     {n_users:,}")
    print(f"  Avg candidates/user:       {avg_cands:.1f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
