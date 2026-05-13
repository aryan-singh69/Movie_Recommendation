"""
preprocess.py — Preprocessing for the Hybrid Movie Recommender System.

This script performs four sequential steps:
  1. Time-aware train/test split (per-user, last 20% → test)
  2. Build content tags for TF-IDF from movie metadata
  3. Compute item-level statistics from training set only (no leakage)
  4. Save processed outputs

No ML or modeling happens here — only preprocessing & feature prep.
"""

import os
import re
import string

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_merged.csv")
RATINGS_PATH = os.path.join(PROCESSED_DIR, "ratings_filtered.csv")


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────
def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the column as a string Series, or empty strings if missing."""
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════
#  STEP 1 — Time-aware train/test split
# ══════════════════════════════════════════════
def time_split(ratings: pd.DataFrame, test_frac: float = 0.20):
    """
    For each user, sort by timestamp and put the last `test_frac`
    of their interactions into the test set. This prevents future-leakage.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Time-aware train/test split")
    print("=" * 60)

    # Sort globally by (userId, timestamp) for deterministic groupby order
    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    # groupby user → split each user's timeline
    print("\n  Splitting per user (this may take a minute on 20M rows) ...")
    for uid, group in ratings.groupby("userId", sort=False):
        n = len(group)
        split_idx = int(n * (1 - test_frac))
        # Ensure at least 1 rating in train
        split_idx = max(split_idx, 1)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    # Sanity checks
    train_users = set(train["userId"].unique())
    test_users = set(test["userId"].unique())
    test_only = test_users - train_users

    print(f"\n  ✓ Train size:  {len(train):>12,}  ({train['userId'].nunique():,} users)")
    print(f"  ✓ Test  size:  {len(test):>12,}  ({test['userId'].nunique():,} users)")
    print(f"  Users in test but NOT in train: {len(test_only)}")
    if len(test_only) > 0:
        print(f"  ⚠ {len(test_only)} test-only users found (they had only 1 rating)")

    return train, test


# ══════════════════════════════════════════════
#  STEP 2 — Build content tags for TF-IDF
# ══════════════════════════════════════════════
def build_tags(movies: pd.DataFrame) -> pd.DataFrame:
    """
    For each movie create a single `tags` string by concatenating:
      - genres (pipes/commas → spaces)
      - overview (first 80 words)
      - cast (top 5 names, spaces → underscores)
      - director (spaces → underscores)
      - keywords (first 10, if column exists)
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Building content tags for TF-IDF")
    print("=" * 60)

    tags_list = []

    # Prefer TMDB genres (richer), fall back to ML genres
    genres_col = "genres_tmdb" if "genres_tmdb" in movies.columns else "genres_ml"
    print(f"\n  Using genres column: {genres_col}")

    for _, row in movies.iterrows():
        parts = []

        # --- Genres ---
        raw_genres = str(row.get(genres_col, "")) if pd.notna(row.get(genres_col)) else ""
        # Replace pipes and commas with spaces
        genres_clean = re.sub(r"[|,]", " ", raw_genres)
        parts.append(genres_clean)

        # --- Overview (first 80 words) ---
        overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""
        overview_words = overview.split()[:80]
        parts.append(" ".join(overview_words))

        # --- Cast (top 5, underscored) ---
        cast_raw = str(row.get("cast", "")) if pd.notna(row.get("cast")) else ""
        if cast_raw:
            # Cast is usually comma-separated names
            cast_names = [name.strip() for name in cast_raw.split(",")][:5]
            cast_names = [name.replace(" ", "_") for name in cast_names]
            parts.append(" ".join(cast_names))

        # --- Director (underscored) ---
        director = str(row.get("director", "")) if pd.notna(row.get("director")) else ""
        if director:
            # Could be comma-separated if multiple directors
            dirs = [d.strip().replace(" ", "_") for d in director.split(",")]
            parts.append(" ".join(dirs))

        # --- Keywords (first 10, if available) ---
        keywords_raw = str(row.get("keywords", "")) if pd.notna(row.get("keywords")) else ""
        if keywords_raw:
            kws = [kw.strip() for kw in keywords_raw.split(",")][:10]
            parts.append(" ".join(kws))

        # Combine & clean
        tag_str = clean_text(" ".join(parts))
        tags_list.append(tag_str)

    movies = movies.copy()
    movies["tags"] = tags_list

    # Stats
    non_empty = (movies["tags"].str.len() > 0).sum()
    print(f"\n  ✓ Movies with non-empty tags: {non_empty} / {len(movies)}")

    # Show 3 samples
    samples = movies[movies["tags"].str.len() > 0].sample(n=min(3, non_empty), random_state=42)
    for _, s in samples.iterrows():
        title = s.get("title_ml", s.get("title_tmdb", "???"))
        print(f"\n  [{title}]")
        print(f"    {s['tags'][:200]}{'…' if len(s['tags']) > 200 else ''}")

    return movies


# ══════════════════════════════════════════════
#  STEP 3 — Compute item-level statistics
#            (from TRAIN set only — no leakage)
# ══════════════════════════════════════════════
def compute_item_stats(movies: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    From the training ratings compute per-movie:
      - global_avg_rating  (mean rating)
      - rating_count       (number of train ratings)
      - rating_count_log   (log1p of count — useful as a popularity feature)
    Merge onto movies and fill missing with 0.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Computing item-level statistics (train only)")
    print("=" * 60)

    stats = (
        train.groupby("movieId")["rating"]
        .agg(global_avg_rating="mean", rating_count="count")
        .reset_index()
    )
    stats["rating_count_log"] = np.log1p(stats["rating_count"])

    print(f"\n  Computed stats for {len(stats):,} movies in training set")
    print(f"  Avg rating range: {stats['global_avg_rating'].min():.2f} – "
          f"{stats['global_avg_rating'].max():.2f}")
    print(f"  Rating count range: {stats['rating_count'].min()} – "
          f"{stats['rating_count'].max():,}")

    # Merge onto movies
    movies = movies.merge(stats, on="movieId", how="left")

    # Fill NaN (movies with 0 training ratings) with 0
    for col in ["global_avg_rating", "rating_count", "rating_count_log"]:
        movies[col] = movies[col].fillna(0)

    print(f"  ✓ Merged item stats → movies shape: {movies.shape}")

    return movies


# ══════════════════════════════════════════════
#  STEP 4 — Save outputs
# ══════════════════════════════════════════════
def save_outputs(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 4 — Saving outputs to data/processed/")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    movies_path = os.path.join(PROCESSED_DIR, "movies_final.csv")

    train.to_csv(train_path, index=False)
    print(f"\n  ✓ Saved {train_path}  ({train.shape})")

    test.to_csv(test_path, index=False)
    print(f"  ✓ Saved {test_path}  ({test.shape})")

    movies.to_csv(movies_path, index=False)
    print(f"  ✓ Saved {movies_path}  ({movies.shape})")

    # Final summary
    print("\n" + "=" * 60)
    print("  DONE — Final Summary")
    print("=" * 60)
    print(f"  Train ratings: {len(train):>12,}")
    print(f"  Test ratings:  {len(test):>12,}")
    print(f"  Movies:        {len(movies):>12,}")
    print(f"  Tag columns:   {'tags' in movies.columns}")
    print(f"  Stat columns:  {[c for c in movies.columns if c in ['global_avg_rating', 'rating_count', 'rating_count_log']]}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════
def main():
    print("\n  Loading data from data/processed/ ...")
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    ratings = pd.read_csv(RATINGS_PATH)
    print(f"  ✓ movies_merged:     {movies.shape}")
    print(f"  ✓ ratings_filtered:  {ratings.shape}")

    # Step 1 — split
    train, test = time_split(ratings)

    # Step 2 — tags
    movies = build_tags(movies)

    # Step 3 — item stats (train only)
    movies = compute_item_stats(movies, train)

    # Step 4 — save
    save_outputs(train, test, movies)


if __name__ == "__main__":
    main()
