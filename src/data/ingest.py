"""
ingest.py — Data ingestion & merging for the Hybrid Movie Recommender System.

This script performs five sequential steps:
  1. Load MovieLens 25M data (ratings, movies, links)
  2. Load the TMDB dataset
  3. Merge ML ↔ TMDB on tmdbId (primary) and clean_title (fallback)
  4. Filter ratings to matched movies & active users (≥20 ratings)
  5. Save processed outputs to data/processed/

No ML, feature engineering, or modeling happens here — only ETL.
"""

import os
import re
import string
import pandas as pd

# ──────────────────────────────────────────────
# Path configuration (relative to project root)
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ML_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ml-25m")
TMDB_PATH = os.path.join(PROJECT_ROOT, "TMDB_all_movies.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ──────────────────────────────────────────────
# Utility: title-cleaning function
# ──────────────────────────────────────────────
def clean_title(title: str) -> str:
    """
    Lowercase, strip the trailing year in parentheses (e.g. "(1995)"),
    remove all punctuation, and collapse whitespace.
    """
    if not isinstance(title, str):
        return ""
    # Remove trailing year pattern like "(1995)" or "(2003)"
    title = re.sub(r"\(\d{4}\)\s*$", "", title)
    # Lowercase
    title = title.lower()
    # Remove punctuation
    title = title.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_year(title: str):
    """
    Pull the 4-digit year from the trailing parenthetical in a MovieLens title.
    Returns an int or None if no year is found.
    """
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else None


# ══════════════════════════════════════════════
#  STEP 1 — Load MovieLens 25M
# ══════════════════════════════════════════════
def load_movielens():
    print("\n" + "=" * 60)
    print("STEP 1 — Loading MovieLens 25M data")
    print("=" * 60)

    # --- ratings.csv ---
    ratings_path = os.path.join(ML_DIR, "ratings.csv")
    print(f"\n  Loading {ratings_path} ...")
    ratings = pd.read_csv(ratings_path)
    print(f"  ✓ ratings shape: {ratings.shape}")

    # --- movies.csv ---
    movies_path = os.path.join(ML_DIR, "movies.csv")
    print(f"\n  Loading {movies_path} ...")
    movies = pd.read_csv(movies_path)
    print(f"  ✓ movies shape:  {movies.shape}")

    # Extract year and clean_title from the MovieLens title column
    movies["year"] = movies["title"].apply(extract_year)
    movies["clean_title"] = movies["title"].apply(clean_title)

    # --- links.csv ---
    links_path = os.path.join(ML_DIR, "links.csv")
    print(f"\n  Loading {links_path} ...")
    links = pd.read_csv(links_path)
    print(f"  ✓ links shape:   {links.shape}")

    # Merge links onto movies so every ML movie gets an imdbId + tmdbId
    movies = movies.merge(links, on="movieId", how="left")
    print(f"\n  After merging links → movies shape: {movies.shape}")
    print(f"  Movies with tmdbId: {movies['tmdbId'].notna().sum()}")

    return ratings, movies


# ══════════════════════════════════════════════
#  STEP 2 — Load TMDB dataset
# ══════════════════════════════════════════════
def load_tmdb():
    print("\n" + "=" * 60)
    print("STEP 2 — Loading TMDB dataset")
    print("=" * 60)

    # The TMDB CSV has some non-UTF-8 characters → use latin-1
    print(f"\n  Loading {TMDB_PATH} ...")
    tmdb = pd.read_csv(TMDB_PATH, encoding="latin-1")
    print(f"  ✓ TMDB shape:   {tmdb.shape}")
    print(f"  Columns: {tmdb.columns.tolist()}")

    # Build the same clean_title for fuzzy-free exact matching
    if "title" in tmdb.columns:
        tmdb["clean_title"] = tmdb["title"].apply(clean_title)
    else:
        print("  ⚠ 'title' column not found in TMDB data; clean_title set to empty")
        tmdb["clean_title"] = ""

    return tmdb


# ══════════════════════════════════════════════
#  STEP 3 — Merge ML ↔ TMDB
# ══════════════════════════════════════════════
def merge_datasets(movies: pd.DataFrame, tmdb: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 3 — Merging MovieLens ↔ TMDB")
    print("=" * 60)

    total_ml = len(movies)

    # --------------------------------------------------
    # Strategy 1 (primary): match on tmdbId ↔ TMDB 'id'
    # --------------------------------------------------
    print("\n  Strategy 1 — matching on tmdbId …")

    # Ensure both keys are the same dtype (float → int where possible)
    movies["tmdbId"] = pd.to_numeric(movies["tmdbId"], errors="coerce")
    tmdb["id"] = pd.to_numeric(tmdb["id"], errors="coerce")

    # Inner join: ML.tmdbId == TMDB.id
    matched_id = movies.dropna(subset=["tmdbId"]).merge(
        tmdb,
        left_on="tmdbId",
        right_on="id",
        how="inner",
        suffixes=("_ml", "_tmdb"),
    )
    n_id = len(matched_id)
    print(f"  ✓ Matched via tmdbId: {n_id}")

    # --------------------------------------------------
    # Strategy 2 (fallback): exact match on clean_title
    # --------------------------------------------------
    print("\n  Strategy 2 — fallback matching on clean_title …")

    # Identify ML movies that did NOT match by ID
    matched_movie_ids = set(matched_id["movieId"].unique())
    unmatched = movies[~movies["movieId"].isin(matched_movie_ids)]

    # Only attempt title match for unmatched movies with a non-empty clean_title
    unmatched = unmatched[unmatched["clean_title"].str.len() > 0]
    tmdb_for_title = tmdb[tmdb["clean_title"].str.len() > 0]

    matched_title = unmatched.merge(
        tmdb_for_title,
        on="clean_title",
        how="inner",
        suffixes=("_ml", "_tmdb"),
    )

    # If multiple TMDB rows match the same clean_title, keep the first
    matched_title = matched_title.drop_duplicates(subset=["movieId"])
    n_title = len(matched_title)
    print(f"  ✓ Matched via clean_title: {n_title}")

    # --------------------------------------------------
    # Combine both matched sets & deduplicate
    # --------------------------------------------------
    merged = pd.concat([matched_id, matched_title], ignore_index=True)
    merged = merged.drop_duplicates(subset=["movieId"])
    n_total = len(merged)

    print(f"\n  ── Merge summary ──")
    print(f"  Total ML movies:           {total_ml}")
    print(f"  Matched via tmdbId:        {n_id}")
    print(f"  Matched via clean_title:   {n_title}")
    print(f"  Total matched (deduped):   {n_total}")
    print(f"  Unmatched ML movies:       {total_ml - n_total}")

    return merged


# ══════════════════════════════════════════════
#  STEP 4 — Filter ratings
# ══════════════════════════════════════════════
def filter_ratings(ratings: pd.DataFrame, valid_movie_ids: set):
    print("\n" + "=" * 60)
    print("STEP 4 — Filtering ratings")
    print("=" * 60)

    initial = len(ratings)
    print(f"\n  Raw ratings count: {initial}")

    # Keep only ratings for movies that survived the merge
    ratings = ratings[ratings["movieId"].isin(valid_movie_ids)].copy()
    print(f"  After keeping matched movies: {len(ratings)}")

    # Keep only users who rated ≥ 20 movies (remove inactive / cold-start users)
    user_counts = ratings["userId"].value_counts()
    active_users = set(user_counts[user_counts >= 20].index)
    ratings = ratings[ratings["userId"].isin(active_users)].copy()

    print(f"  After removing users with <20 ratings: {len(ratings)}")
    print(f"  Unique active users: {len(active_users)}")

    return ratings


# ══════════════════════════════════════════════
#  STEP 5 — Save processed outputs
# ══════════════════════════════════════════════
def save_outputs(movies_merged: pd.DataFrame, ratings_filtered: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 5 — Saving to data/processed/")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    movies_out = os.path.join(OUTPUT_DIR, "movies_merged.csv")
    ratings_out = os.path.join(OUTPUT_DIR, "ratings_filtered.csv")

    movies_merged.to_csv(movies_out, index=False)
    print(f"\n  ✓ Saved {movies_out}  ({movies_merged.shape})")

    ratings_filtered.to_csv(ratings_out, index=False)
    print(f"  ✓ Saved {ratings_out}  ({ratings_filtered.shape})")

    # Final summary
    print("\n" + "=" * 60)
    print("  DONE — Final Summary")
    print("=" * 60)
    print(f"  Movies:  {len(movies_merged):>10,}")
    print(f"  Ratings: {len(ratings_filtered):>10,}")
    print(f"  Users:   {ratings_filtered['userId'].nunique():>10,}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════
def main():
    # Step 1
    ratings, movies = load_movielens()

    # Step 2
    tmdb = load_tmdb()

    # Step 3
    movies_merged = merge_datasets(movies, tmdb)

    # Step 4
    valid_ids = set(movies_merged["movieId"].unique())
    ratings_filtered = filter_ratings(ratings, valid_ids)

    # Step 5
    save_outputs(movies_merged, ratings_filtered)


if __name__ == "__main__":
    main()
