"""
rerank.py — Post-scoring re-ranking filters.

Applies business-rule filters on top of scored candidate lists:
  1. Remove movies the user already rated (train set)
  2. Remove movies with global_avg_rating < 3.5
  3. Cap same primary genre at max 3 movies per user
  4. Return top 10 after all filters

Also provides rerank_batch() to apply the above across all users.
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MIN_GLOBAL_RATING = 3.5     # Minimum global average rating to keep
MAX_PER_GENRE = 3           # Max movies from the same primary genre
TOP_K = 10                  # Final list size


# ══════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════
def _get_primary_genre(movie_id: int, genre_lookup: dict) -> str:
    """Return the first (primary) genre for a movie, or 'Unknown'."""
    genres = genre_lookup.get(movie_id, [])
    return genres[0] if genres else "Unknown"


def _parse_genres(genre_str) -> list:
    """Split genre string (pipe or comma separated) into a list."""
    if not isinstance(genre_str, str) or genre_str.strip() == "":
        return []
    if "|" in genre_str:
        return [g.strip() for g in genre_str.split("|") if g.strip()]
    return [g.strip() for g in genre_str.split(",") if g.strip()]


def _build_lookups(movies_df: pd.DataFrame):
    """
    Pre-build lookup dicts from the movies dataframe:
      - rating_lookup: movieId → global_avg_rating
      - genre_lookup:  movieId → [genre1, genre2, ...]
    """
    rating_lookup = dict(
        zip(movies_df["movieId"].values, movies_df["global_avg_rating"].values)
    )

    genre_col = "genres_ml" if "genres_ml" in movies_df.columns else "genres_tmdb"
    genre_lookup = {}
    for mid, gstr in zip(movies_df["movieId"].values, movies_df[genre_col].values):
        genre_lookup[mid] = _parse_genres(gstr)

    return rating_lookup, genre_lookup


# ══════════════════════════════════════════════
#  Core rerank function
# ══════════════════════════════════════════════
def rerank(recommendations, user_watched, movies_df,
           _rating_lookup=None, _genre_lookup=None):
    """
    Apply business-rule re-ranking filters to a single user's
    candidate list.

    Parameters
    ----------
    recommendations : list of (movieId, score)
        Candidates sorted by score descending.
    user_watched : set of int
        movieIds the user already rated in the training set.
    movies_df : pd.DataFrame
        movies_final.csv loaded as a dataframe.
    _rating_lookup : dict, optional
        Pre-built movieId → global_avg_rating (for batch mode).
    _genre_lookup : dict, optional
        Pre-built movieId → [genre, ...] (for batch mode).

    Returns
    -------
    list of int
        Top-10 movieIds after filtering.
    """
    # Build lookups on-the-fly if not provided (single-user mode)
    if _rating_lookup is None or _genre_lookup is None:
        _rating_lookup, _genre_lookup = _build_lookups(movies_df)

    selected = []
    genre_counts = {}  # primary_genre → count of selected movies

    for movie_id, score in recommendations:
        # ── Rule 1: skip already-watched ──
        if movie_id in user_watched:
            continue

        # ── Rule 2: skip low global rating ──
        global_rating = _rating_lookup.get(movie_id, 0.0)
        if global_rating < MIN_GLOBAL_RATING:
            continue

        # ── Rule 3: cap same primary genre at MAX_PER_GENRE ──
        primary_genre = _get_primary_genre(movie_id, _genre_lookup)
        if genre_counts.get(primary_genre, 0) >= MAX_PER_GENRE:
            continue

        # Passed all filters — select this movie
        selected.append(movie_id)
        genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1

        # ── Rule 4: stop at TOP_K ──
        if len(selected) >= TOP_K:
            break

    return selected


# ══════════════════════════════════════════════
#  Batch rerank for all users
# ══════════════════════════════════════════════
def rerank_batch(candidates_dict, train_df, movies_df):
    """
    Apply rerank() to every user in the candidates dict.

    Parameters
    ----------
    candidates_dict : dict
        {userId: [(movieId, score), ...]} — scored candidates per user,
        sorted by score descending.
    train_df : pd.DataFrame
        Training ratings (userId, movieId, rating).
    movies_df : pd.DataFrame
        movies_final.csv loaded as a dataframe.

    Returns
    -------
    dict
        {userId: [movieId, movieId, ...]} — final top-10 list per user.
    """
    print("\n" + "=" * 60)
    print("RE-RANKING — Applying business-rule filters")
    print("=" * 60)

    # Pre-build lookups once (shared across all users)
    rating_lookup, genre_lookup = _build_lookups(movies_df)

    # Pre-build per-user watched sets from training data
    print("\n  Building per-user watched sets ...")
    user_watched_map = {}
    for uid, mid in zip(train_df["userId"].values, train_df["movieId"].values):
        if uid not in user_watched_map:
            user_watched_map[uid] = set()
        user_watched_map[uid].add(mid)

    print(f"  Users in train: {len(user_watched_map):,}")
    print(f"  Users with candidates: {len(candidates_dict):,}")

    # Apply rerank to each user
    print("\n  Re-ranking all users ...")
    t0 = time.time()
    final_recs = {}
    total_users = len(candidates_dict)

    for count, (uid, recs) in enumerate(candidates_dict.items(), 1):
        watched = user_watched_map.get(uid, set())
        final_recs[uid] = rerank(
            recs, watched, movies_df,
            _rating_lookup=rating_lookup,
            _genre_lookup=genre_lookup,
        )

        if count % 25_000 == 0:
            elapsed = time.time() - t0
            print(f"    ... processed {count:>8,} / {total_users:,} "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\n  ✓ Re-ranking completed in {elapsed:.1f}s")

    # ── Summary stats ──
    list_lengths = [len(v) for v in final_recs.values()]
    avg_len = np.mean(list_lengths) if list_lengths else 0.0
    full_10 = sum(1 for l in list_lengths if l == TOP_K)

    print(f"\n  ── Re-rank Summary ──")
    print(f"  Total users processed:           {len(final_recs):,}")
    print(f"  Avg final list length:           {avg_len:.2f}")
    print(f"  Users with full {TOP_K} recs:       {full_10:,} "
          f"({full_10 / max(len(final_recs), 1) * 100:.1f}%)")
    print(f"  Users with 0 recs:               "
          f"{sum(1 for l in list_lengths if l == 0):,}")
    print("=" * 60)

    return final_recs


# ══════════════════════════════════════════════
#  Standalone entry point
# ══════════════════════════════════════════════
def main():
    """Load artifacts, run batch rerank, and save results."""
    print("\n  Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    print(f"  ✓ train:  {train.shape}")
    print(f"  ✓ movies: {movies.shape}")

    # Load scored candidates (from ranker or raw CF/content)
    # Try LightGBM-scored candidates first, fall back to raw CF candidates
    scored_path = os.path.join(ARTIFACTS_DIR, "scored_candidates.pkl")
    cf_path = os.path.join(ARTIFACTS_DIR, "cf_candidates.pkl")

    if os.path.exists(scored_path):
        print("\n  Loading scored candidates (LightGBM) ...")
        with open(scored_path, "rb") as f:
            candidates = pickle.load(f)
        print(f"  ✓ Scored candidates: {len(candidates):,} users")
    elif os.path.exists(cf_path):
        print("\n  Loading raw CF candidates (no ranker scores found) ...")
        with open(cf_path, "rb") as f:
            candidates = pickle.load(f)
        print(f"  ✓ CF candidates: {len(candidates):,} users")
    else:
        raise FileNotFoundError(
            "No candidate files found. Run cf_candidates.py or train_ranker.py first."
        )

    # Run batch rerank
    final_recs = rerank_batch(candidates, train, movies)

    # Save final recommendations
    output_path = os.path.join(ARTIFACTS_DIR, "final_recommendations.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(final_recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  ✓ Saved {output_path} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("  DONE — Re-ranking complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
