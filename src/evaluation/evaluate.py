"""
evaluate.py — Evaluate 4 recommendation models and print a comparison table.

Models evaluated:
  1. Popularity baseline   — top-10 globally most-rated movies for every user
  2. Content-only          — top-10 from content_candidates sorted by content_score
  3. CF-only               — top-10 from cf_candidates sorted by cf_score
  4. Hybrid + LightGBM     — merged candidates scored by the ranker, then reranked

All 4 models are post-processed by rerank() (remove watched, quality filter,
diversity cap).  Metrics are evaluated on a 5,000-user sample for speed.

Usage:
    python -m src.evaluation.evaluate
    python src/evaluation/evaluate.py
"""

import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")

CF_CAND_PATH = os.path.join(ARTIFACTS_DIR, "cf_candidates.pkl")
CONTENT_CAND_PATH = os.path.join(ARTIFACTS_DIR, "content_candidates.pkl")
RANKER_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_ranker.pkl")
USER_FEAT_PATH = os.path.join(ARTIFACTS_DIR, "user_features.pkl")
FEAT_COLS_PATH = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
K = 10
SAMPLE_USERS = 5_000
POSITIVE_THRESHOLD = 4.0
RANDOM_SEED = 42

# Allow imports from project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import (
    coverage,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.ranker.rerank import (
    _build_lookups,
    _parse_genres,
    rerank,
)


# ══════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════
def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_genre_lookup(movies_df: pd.DataFrame) -> dict:
    """movieId → [genre1, genre2, ...]"""
    genre_col = "genres_ml" if "genres_ml" in movies_df.columns else "genres_tmdb"
    lookup = {}
    for mid, gstr in zip(movies_df["movieId"].values, movies_df[genre_col].values):
        lookup[mid] = _parse_genres(gstr)
    return lookup


def _build_user_watched(train_df: pd.DataFrame) -> dict:
    """userId → set of movieIds rated in train."""
    watched = defaultdict(set)
    for u, m in zip(train_df["userId"].values, train_df["movieId"].values):
        watched[u].add(m)
    return dict(watched)


def _build_test_relevant(test_df: pd.DataFrame, threshold: float) -> dict:
    """userId → set of movieIds rated ≥ threshold in test."""
    relevant = defaultdict(set)
    for u, m, r in zip(
        test_df["userId"].values,
        test_df["movieId"].values,
        test_df["rating"].values,
    ):
        if r >= threshold:
            relevant[u].add(m)
    return dict(relevant)


# ══════════════════════════════════════════════
#  Model generators
# ══════════════════════════════════════════════
def _popularity_candidates(train_df: pd.DataFrame, n: int = 200) -> list:
    """
    Return the top-n globally most-rated movies (by count) as
    [(movieId, score), ...] where score = rating_count so that
    rerank sees the same interface as other candidate lists.
    """
    counts = train_df.groupby("movieId").size().reset_index(name="cnt")
    counts = counts.sort_values("cnt", ascending=False).head(n)
    return list(zip(counts["movieId"].values, counts["cnt"].values.astype(float)))


def _score_hybrid(
    user_id,
    cf_cands: dict,
    content_cands: dict,
    ranker,
    user_features_df: pd.DataFrame,
    feature_cols: list,
    movie_lookup: dict,
    uf_lookup,
):
    """
    Merge CF + content candidates for a user, build feature vectors,
    score with the LightGBM ranker, and return [(movieId, score), ...]
    sorted by score descending.
    """
    # Merge candidate scores
    movie_scores = {}
    for mid, score in cf_cands.get(user_id, []):
        movie_scores[mid] = {"cf_score": score, "content_score": 0.0}
    for mid, score in content_cands.get(user_id, []):
        if mid in movie_scores:
            movie_scores[mid]["content_score"] = score
        else:
            movie_scores[mid] = {"cf_score": 0.0, "content_score": score}

    if not movie_scores:
        return []

    # User features
    if user_id in uf_lookup.index:
        uf = uf_lookup.loc[user_id]
        u_avg = float(uf["avg_rating_given"])
        u_count = float(uf["rating_count"])
        u_entropy = float(uf["genre_entropy"])
    else:
        u_avg, u_count, u_entropy = 0.0, 0.0, 0.0

    # Build feature matrix
    mids = []
    rows = []
    for mid, scores in movie_scores.items():
        mf = movie_lookup.get(mid, {})
        row = [
            scores.get("cf_score", 0.0),
            scores.get("content_score", 0.0),
            mf.get("global_avg_rating", 0.0),
            mf.get("rating_count_log", 0.0),
            u_avg,
            u_count,
            u_entropy,
        ]
        rows.append(row)
        mids.append(mid)

    X = pd.DataFrame(rows, columns=feature_cols, dtype=np.float32)
    probs = ranker.predict_proba(X)[:, 1]

    # Sort by predicted probability descending
    order = np.argsort(-probs)
    return [(mids[i], float(probs[i])) for i in order]


# ══════════════════════════════════════════════
#  Per-model evaluation
# ══════════════════════════════════════════════
def evaluate_model(
    model_name: str,
    user_ids: list,
    candidate_fn,
    user_watched: dict,
    user_relevant: dict,
    movies_df: pd.DataFrame,
    genre_lookup: dict,
    rating_lookup: dict,
    total_catalog_size: int,
):
    """
    Run rerank + metric computation for one model over the sampled users.

    Parameters
    ----------
    candidate_fn : callable(user_id) -> list of (movieId, score)
        Returns raw candidates for a single user.

    Returns
    -------
    dict  with keys: P@10, R@10, NDCG@10, Coverage, Diversity
    """
    all_recs = {}
    precisions, recalls, ndcgs, diversities = [], [], [], []

    for uid in user_ids:
        raw_candidates = candidate_fn(uid)
        watched = user_watched.get(uid, set())
        rel = user_relevant.get(uid, set())

        # Apply rerank (remove watched, quality filter, diversity cap)
        final_list = rerank(
            raw_candidates,
            watched,
            movies_df,
            _rating_lookup=rating_lookup,
            _genre_lookup=genre_lookup,
        )

        all_recs[uid] = final_list

        # Skip users with no relevant items for per-user metrics
        if len(rel) == 0:
            continue

        precisions.append(precision_at_k(final_list, rel, K))
        recalls.append(recall_at_k(final_list, rel, K))
        ndcgs.append(ndcg_at_k(final_list, rel, K))
        diversities.append(intra_list_diversity(final_list, genre_lookup))

    # Aggregate
    cov = coverage(all_recs, total_catalog_size)
    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    avg_rec = float(np.mean(recalls)) if recalls else 0.0
    avg_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    avg_div = float(np.mean(diversities)) if diversities else 0.0

    return {
        "Model": model_name,
        "P@10": avg_prec,
        "R@10": avg_rec,
        "NDCG@10": avg_ndcg,
        "Coverage": cov,
        "Diversity": avg_div,
    }


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════
def main():
    t_start = time.time()

    print("\n" + "=" * 65)
    print("  EVALUATION — Comparing 4 recommendation models")
    print("=" * 65)

    # ── Load data ──────────────────────────────
    print("\n  Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    print(f"  ✓ train:  {train.shape}")
    print(f"  ✓ test:   {test.shape}")
    print(f"  ✓ movies: {movies.shape}")

    print("\n  Loading artifacts ...")
    cf_cands = _load_pkl(CF_CAND_PATH)
    content_cands = _load_pkl(CONTENT_CAND_PATH)
    ranker = _load_pkl(RANKER_PATH)
    user_features = _load_pkl(USER_FEAT_PATH)
    feature_cols = _load_pkl(FEAT_COLS_PATH)
    print(f"  ✓ CF candidates:      {len(cf_cands):,} users")
    print(f"  ✓ Content candidates:  {len(content_cands):,} users")
    print(f"  ✓ LightGBM ranker loaded")
    print(f"  ✓ Feature columns: {feature_cols}")

    # ── Build lookups ──────────────────────────
    print("\n  Building lookups ...")
    user_watched = _build_user_watched(train)
    user_relevant = _build_test_relevant(test, POSITIVE_THRESHOLD)
    rating_lookup, genre_lookup_rerank = _build_lookups(movies)
    genre_lookup = _build_genre_lookup(movies)
    total_catalog = movies["movieId"].nunique()
    print(f"  ✓ Catalog size: {total_catalog:,}")
    print(f"  ✓ Users with watched history: {len(user_watched):,}")
    print(f"  ✓ Users with relevant items:  {len(user_relevant):,}")

    # ── Movie feature lookup for hybrid scoring ──
    movie_lookup = movies.set_index("movieId")[
        ["global_avg_rating", "rating_count_log"]
    ].to_dict("index")
    uf_lookup = user_features.set_index("userId")

    # ── Sample test users ──────────────────────
    test_users = sorted(set(test["userId"].unique()))
    rng = np.random.RandomState(RANDOM_SEED)
    if len(test_users) > SAMPLE_USERS:
        sample_users = list(rng.choice(test_users, size=SAMPLE_USERS, replace=False))
    else:
        sample_users = test_users
    print(f"\n  Evaluating on {len(sample_users):,} sampled users (K={K})")

    # ── Popularity baseline candidates ─────────
    pop_candidates = _popularity_candidates(train, n=200)

    # ── Evaluate each model ────────────────────
    results = []

    # --- 1. Popularity baseline ---
    print("\n  [1/4] Evaluating Popularity baseline ...")
    t0 = time.time()
    res = evaluate_model(
        model_name="Popularity",
        user_ids=sample_users,
        candidate_fn=lambda uid: pop_candidates,
        user_watched=user_watched,
        user_relevant=user_relevant,
        movies_df=movies,
        genre_lookup=genre_lookup,
        rating_lookup=rating_lookup,
        total_catalog_size=total_catalog,
    )
    print(f"        done in {time.time() - t0:.1f}s")
    results.append(res)

    # --- 2. Content-only ---
    print("  [2/4] Evaluating Content-only ...")
    t0 = time.time()
    res = evaluate_model(
        model_name="Content-only",
        user_ids=sample_users,
        candidate_fn=lambda uid: content_cands.get(uid, []),
        user_watched=user_watched,
        user_relevant=user_relevant,
        movies_df=movies,
        genre_lookup=genre_lookup,
        rating_lookup=rating_lookup,
        total_catalog_size=total_catalog,
    )
    print(f"        done in {time.time() - t0:.1f}s")
    results.append(res)

    # --- 3. CF-only ---
    print("  [3/4] Evaluating CF-only ...")
    t0 = time.time()
    res = evaluate_model(
        model_name="CF-only",
        user_ids=sample_users,
        candidate_fn=lambda uid: cf_cands.get(uid, []),
        user_watched=user_watched,
        user_relevant=user_relevant,
        movies_df=movies,
        genre_lookup=genre_lookup,
        rating_lookup=rating_lookup,
        total_catalog_size=total_catalog,
    )
    print(f"        done in {time.time() - t0:.1f}s")
    results.append(res)

    # --- 4. Hybrid + LightGBM ---
    print("  [4/4] Evaluating Hybrid+LightGBM ...")
    t0 = time.time()
    res = evaluate_model(
        model_name="Hybrid+LightGBM",
        user_ids=sample_users,
        candidate_fn=lambda uid: _score_hybrid(
            uid, cf_cands, content_cands,
            ranker, user_features, feature_cols,
            movie_lookup, uf_lookup,
        ),
        user_watched=user_watched,
        user_relevant=user_relevant,
        movies_df=movies,
        genre_lookup=genre_lookup,
        rating_lookup=rating_lookup,
        total_catalog_size=total_catalog,
    )
    print(f"        done in {time.time() - t0:.1f}s")
    results.append(res)

    # ── Build results table ────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "P@10", "R@10", "NDCG@10", "Coverage", "Diversity"]]

    # ── Print table ────────────────────────────
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    header = (
        f"{'Model':<20s} {'P@10':>7s} {'R@10':>7s} {'NDCG@10':>8s} "
        f"{'Coverage':>9s} {'Diversity':>10s}"
    )
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        line = (
            f"{row['Model']:<20s} {row['P@10']:>7.3f} {row['R@10']:>7.3f} "
            f"{row['NDCG@10']:>8.3f} {row['Coverage']:>9.3f} "
            f"{row['Diversity']:>10.3f}"
        )
        print(line)
    print("=" * 65)

    # ── Save to CSV ────────────────────────────
    output_path = os.path.join(ARTIFACTS_DIR, "evaluation_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved results to {output_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total evaluation time: {elapsed:.1f}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
