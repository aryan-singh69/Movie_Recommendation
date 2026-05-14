"""
train_ranker.py — Build features and train a LightGBM ranking model.

Pipeline:
  1. Build user features from train.csv (no test leakage)
  2. Merge CF + content candidates per user
  3. Build (user, movie) training pairs with labels from test.csv
  4. Train LightGBM classifier on sampled users
  5. Save model + feature artifacts

No serving logic here — only offline training.
"""

import gc
import os
import pickle
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_final.csv")
CF_CAND_PATH = os.path.join(ARTIFACTS_DIR, "cf_candidates.pkl")
CONT_CAND_PATH = os.path.join(ARTIFACTS_DIR, "content_candidates.pkl")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MAX_USERS = 10_000          # Sample size for training (memory constraint)
POSITIVE_THRESHOLD = 4.0    # rating ≥ 4.0 in test → positive label


# ══════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════
def _load_pkl(name):
    path = os.path.join(ARTIFACTS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pkl(obj, name):
    path = os.path.join(ARTIFACTS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✓ Saved {name} ({size_mb:.1f} MB)")


def _parse_genres(genre_str):
    """Split genre string (pipe or comma separated) into a list."""
    if not isinstance(genre_str, str) or genre_str.strip() == "":
        return []
    # Handle both ML (pipe) and TMDB (comma) formats
    if "|" in genre_str:
        return [g.strip() for g in genre_str.split("|") if g.strip()]
    return [g.strip() for g in genre_str.split(",") if g.strip()]


# ══════════════════════════════════════════════
#  STEP 1 — Build user features (train only)
# ══════════════════════════════════════════════
def build_user_features(train: pd.DataFrame, movies: pd.DataFrame):
    """
    From training ratings compute per-user:
      - avg_rating_given: mean of all ratings
      - rating_count: total number of ratings
      - genre_entropy: Shannon entropy of genre distribution
      - top_genre: most frequently rated genre (label encoded)
    """
    print("\n" + "=" * 60)
    print("STEP 1 — Building user features (train only)")
    print("=" * 60)

    # Basic stats
    user_stats = (
        train.groupby("userId")["rating"]
        .agg(avg_rating_given="mean", rating_count="count")
        .reset_index()
    )

    # Build movieId → genre list lookup
    # Prefer genres_ml (pipe-separated, cleaner for ML categories)
    genre_col = "genres_ml" if "genres_ml" in movies.columns else "genres_tmdb"
    movie_genres = {}
    for _, row in movies.iterrows():
        movie_genres[row["movieId"]] = _parse_genres(str(row.get(genre_col, "")))

    # Per-user genre distribution
    print("\n  Computing genre distributions ...")
    user_genre_counts = defaultdict(Counter)
    for uid, mid in zip(train["userId"].values, train["movieId"].values):
        genres = movie_genres.get(mid, [])
        for g in genres:
            user_genre_counts[uid][g] += 1

    # Compute entropy + top genre
    entropies = []
    top_genres = []
    for uid in user_stats["userId"]:
        counts = user_genre_counts.get(uid, Counter())
        total = sum(counts.values())
        if total == 0:
            entropies.append(0.0)
            top_genres.append("Unknown")
            continue
        probs = np.array(list(counts.values()), dtype=np.float64) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(float(entropy))
        top_genres.append(counts.most_common(1)[0][0])

    user_stats["genre_entropy"] = entropies
    user_stats["top_genre_raw"] = top_genres

    # Label-encode top_genre
    le = LabelEncoder()
    user_stats["top_genre"] = le.fit_transform(user_stats["top_genre_raw"])

    print(f"  ✓ User features shape: {user_stats.shape}")
    print(f"  Unique top genres: {len(le.classes_)}")
    print(f"  Avg rating given: {user_stats['avg_rating_given'].mean():.2f}")
    print(f"  Avg genre entropy: {user_stats['genre_entropy'].mean():.2f}")

    return user_stats, le


# ══════════════════════════════════════════════
#  STEP 2 — Merge candidates
# ══════════════════════════════════════════════
def merge_candidates(cf_cands: dict, content_cands: dict, train: pd.DataFrame):
    """
    For each user: union CF + content candidates, deduplicate,
    remove already-rated movies.
    Returns: {userId: {movieId: {'cf_score': float, 'content_score': float}}}
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Merging CF + content candidates")
    print("=" * 60)

    # Precompute per-user rated sets from train
    print("\n  Building rated-movie lookup ...")
    user_rated = defaultdict(set)
    for u, m in zip(train["userId"].values, train["movieId"].values):
        user_rated[u].add(m)

    # All user IDs that have candidates from either source
    all_users = set(cf_cands.keys()) | set(content_cands.keys())
    print(f"  Users with any candidates: {len(all_users):,}")

    merged = {}
    total_cands = 0
    for uid in all_users:
        movie_scores = {}

        # CF candidates
        for mid, score in cf_cands.get(uid, []):
            movie_scores[mid] = {"cf_score": score, "content_score": 0.0}

        # Content candidates — merge, keeping max if duplicate
        for mid, score in content_cands.get(uid, []):
            if mid in movie_scores:
                movie_scores[mid]["content_score"] = score
            else:
                movie_scores[mid] = {"cf_score": 0.0, "content_score": score}

        # Remove already-rated
        rated = user_rated.get(uid, set())
        movie_scores = {m: s for m, s in movie_scores.items() if m not in rated}

        if movie_scores:
            merged[uid] = movie_scores
            total_cands += len(movie_scores)

    avg_cands = total_cands / max(len(merged), 1)
    print(f"  ✓ Merged candidates for {len(merged):,} users")
    print(f"  Avg candidates/user: {avg_cands:.1f}")

    return merged


# ══════════════════════════════════════════════
#  STEP 3 — Build training pairs
# ══════════════════════════════════════════════
def build_training_data(merged_cands: dict, test: pd.DataFrame,
                        user_features: pd.DataFrame,
                        movies: pd.DataFrame):
    """
    For each (user, candidate_movie) pair build a feature vector
    and assign label 1 if user rated movie ≥ 4.0 in test, else 0.
    Sample up to MAX_USERS users for memory.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Building training pairs")
    print("=" * 60)

    # Build test lookup: {(userId, movieId): rating}
    print("\n  Building test lookup ...")
    test_lookup = {}
    for u, m, r in zip(test["userId"].values, test["movieId"].values, test["rating"].values):
        test_lookup[(u, m)] = r

    # Only keep users that have ≥ 1 positive label in test
    # (among their candidate movies)
    print("  Filtering users with ≥1 positive test label ...")
    eligible_users = []
    for uid, cands in merged_cands.items():
        has_positive = False
        for mid in cands:
            rating = test_lookup.get((uid, mid))
            if rating is not None and rating >= POSITIVE_THRESHOLD:
                has_positive = True
                break
        if has_positive:
            eligible_users.append(uid)

    print(f"  Eligible users (≥1 positive): {len(eligible_users):,}")

    # Sample if too many
    if len(eligible_users) > MAX_USERS:
        rng = np.random.RandomState(42)
        eligible_users = list(rng.choice(eligible_users, size=MAX_USERS, replace=False))
        print(f"  Sampled to {MAX_USERS:,} users")

    # Build user feature lookup (indexed by userId)
    uf_lookup = user_features.set_index("userId")

    # Build movie feature lookup
    movie_lookup = movies.set_index("movieId")[
        ["global_avg_rating", "rating_count_log"]
    ].to_dict("index")

    # Feature columns in order
    feature_cols = [
        "cf_score",
        "content_score",
        "global_avg_rating",
        "rating_count_log",
        "avg_rating_given",
        "rating_count",
        "genre_entropy",
    ]

    # Build rows
    print("  Assembling feature matrix ...")
    rows = []
    labels = []

    for uid in eligible_users:
        cands = merged_cands.get(uid, {})
        # User features
        if uid in uf_lookup.index:
            uf = uf_lookup.loc[uid]
            u_avg = float(uf["avg_rating_given"])
            u_count = float(uf["rating_count"])
            u_entropy = float(uf["genre_entropy"])
        else:
            u_avg, u_count, u_entropy = 0.0, 0.0, 0.0

        for mid, scores in cands.items():
            # Movie features
            mf = movie_lookup.get(mid, {})
            g_avg = mf.get("global_avg_rating", 0.0)
            rc_log = mf.get("rating_count_log", 0.0)

            row = [
                scores.get("cf_score", 0.0),
                scores.get("content_score", 0.0),
                g_avg,
                rc_log,
                u_avg,
                u_count,
                u_entropy,
            ]
            rows.append(row)

            # Label from test
            test_rating = test_lookup.get((uid, mid))
            label = 1 if (test_rating is not None and test_rating >= POSITIVE_THRESHOLD) else 0
            labels.append(label)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"\n  ✓ Training matrix shape: {X.shape}")
    print(f"  Class balance — positives: {n_pos:,} ({n_pos / len(y) * 100:.1f}%)")
    print(f"                  negatives: {n_neg:,} ({n_neg / len(y) * 100:.1f}%)")
    print(f"  Feature columns: {feature_cols}")

    return X, y, feature_cols


# ══════════════════════════════════════════════
#  STEP 4 — Train LightGBM
# ══════════════════════════════════════════════
def train_model(X, y, feature_cols):
    """
    Train an LGBMClassifier with 80/20 train/val split.
    Print feature importances and validation AUC.
    """
    print("\n" + "=" * 60)
    print("STEP 4 — Training LightGBM classifier")
    print("=" * 60)

    # 80/20 split stratified by label
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape},  Val: {X_val.shape}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    print("  Training ...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
    )
    elapsed = time.time() - t0
    print(f"  ✓ Training completed in {elapsed:.1f}s")

    # Validation AUC
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"\n  ── Validation AUC: {val_auc:.4f} ──")

    # Feature importances
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n  Feature Importances:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] / max(importances) * 30)
        print(f"    {row['feature']:<22s} {row['importance']:>6d}  {bar}")

    return model


# ══════════════════════════════════════════════
#  STEP 5 — Save artifacts
# ══════════════════════════════════════════════
def save_artifacts(model, user_features, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 5 — Saving artifacts to data/artifacts/")
    print("=" * 60)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    _save_pkl(model, "lgbm_ranker.pkl")
    _save_pkl(user_features, "user_features.pkl")
    _save_pkl(feature_cols, "feature_columns.pkl")


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════
def main():
    print("\n  Loading data ...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH, encoding="latin-1")
    print(f"  ✓ train:  {train.shape}")
    print(f"  ✓ test:   {test.shape}")
    print(f"  ✓ movies: {movies.shape}")

    print("\n  Loading candidate dicts ...")
    cf_cands = _load_pkl("cf_candidates.pkl")
    print(f"  ✓ CF candidates: {len(cf_cands):,} users")
    content_cands = _load_pkl("content_candidates.pkl")
    print(f"  ✓ Content candidates: {len(content_cands):,} users")

    # Step 1 — user features
    user_features, genre_le = build_user_features(train, movies)

    # Free train movie-level data we no longer need
    gc.collect()

    # Step 2 — merge candidates
    merged = merge_candidates(cf_cands, content_cands, train)

    # Free raw candidate dicts
    del cf_cands, content_cands
    gc.collect()

    # Step 3 — build training pairs
    X, y, feature_cols = build_training_data(merged, test, user_features, movies)

    # Free merged candidates
    del merged
    gc.collect()

    # Step 4 — train
    model = train_model(X, y, feature_cols)

    # Step 5 — save
    save_artifacts(model, user_features, feature_cols)

    # Final summary
    print("\n" + "=" * 60)
    print("  DONE — Ranker training complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
