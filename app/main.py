from contextlib import asynccontextmanager
import difflib
import os
import pickle
import sys

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from scipy.sparse import issparse

# Path configuration for ML artifacts
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "data", "artifacts")

# Allow imports from project root for the reranker package.
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ranker.rerank import rerank, _build_lookups


ml_artifacts: dict = {}


def _load_pkl(name: str):
    path = os.path.join(_ARTIFACTS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ml_artifacts():
    """Load all ML artifacts into the global store. Called once at startup."""
    print("\n" + "=" * 60)
    print("  STARTUP - Loading ML artifacts")
    print("=" * 60)

    try:
        ml_artifacts["cf_candidates"] = _load_pkl("cf_candidates.pkl")
        print(f"  [ok] cf_candidates.pkl          ({len(ml_artifacts['cf_candidates']):,} users)")
    except Exception as e:
        print(f"  [fail] cf_candidates.pkl        {e}")

    try:
        ml_artifacts["content_candidates"] = _load_pkl("content_candidates.pkl")
        print(f"  [ok] content_candidates.pkl     ({len(ml_artifacts['content_candidates']):,} users)")
    except Exception as e:
        print(f"  [fail] content_candidates.pkl   {e}")

    try:
        ml_artifacts["ranker"] = _load_pkl("lgbm_ranker.pkl")
        print("  [ok] lgbm_ranker.pkl            loaded")
    except Exception as e:
        print(f"  [fail] lgbm_ranker.pkl          {e}")

    try:
        ml_artifacts["user_features"] = _load_pkl("user_features.pkl")
        print(f"  [ok] user_features.pkl          ({ml_artifacts['user_features'].shape})")
    except Exception as e:
        print(f"  [fail] user_features.pkl        {e}")

    try:
        ml_artifacts["feature_columns"] = _load_pkl("feature_columns.pkl")
        print(f"  [ok] feature_columns.pkl        {ml_artifacts['feature_columns']}")
    except Exception as e:
        print(f"  [fail] feature_columns.pkl      {e}")

    try:
        movies_path = os.path.join(_PROCESSED_DIR, "movies_final.csv")
        ml_artifacts["movies_df"] = pd.read_csv(movies_path, encoding="latin-1")
        print(f"  [ok] movies_final.csv           ({ml_artifacts['movies_df'].shape})")
    except Exception as e:
        print(f"  [fail] movies_final.csv         {e}")

    try:
        ml_artifacts["movie_id_to_idx"] = _load_pkl("movie_id_to_idx.pkl")
        ml_artifacts["idx_to_movie_id"] = _load_pkl("idx_to_movie_id.pkl")
        print(f"  [ok] movie_id_to_idx.pkl        ({len(ml_artifacts['movie_id_to_idx']):,} movies)")
    except Exception as e:
        print(f"  [fail] movie_id_to_idx.pkl      {e}")

    try:
        ml_artifacts["tfidf_matrix"] = _load_pkl("tfidf_matrix.pkl")
        mat = ml_artifacts["tfidf_matrix"]
        dense = mat.toarray().astype(np.float32) if issparse(mat) else np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_normed = dense / norms

        index = faiss.IndexFlatIP(dense_normed.shape[1])
        index.add(dense_normed)

        ml_artifacts["faiss_index"] = index
        ml_artifacts["dense_normed"] = dense_normed
        print(f"  [ok] tfidf_matrix.pkl           shape={mat.shape}")
        print(f"  [ok] FAISS index rebuilt        ({index.ntotal:,} vectors)")
    except Exception as e:
        print(f"  [fail] tfidf_matrix / FAISS     {e}")

    if "movies_df" in ml_artifacts:
        movies = ml_artifacts["movies_df"]
        rating_lk, genre_lk = _build_lookups(movies)
        ml_artifacts["rating_lookup"] = rating_lk
        ml_artifacts["genre_lookup"] = genre_lk

        ml_artifacts["movie_info"] = {}
        genre_col = "genres_ml" if "genres_ml" in movies.columns else "genres_tmdb"
        for _, row in movies.iterrows():
            ml_artifacts["movie_info"][row["movieId"]] = {
                "movieId": int(row["movieId"]),
                "title": str(row.get("title_ml", row.get("title_tmdb", "Unknown"))),
                "genres": str(row.get(genre_col, "")),
                "global_avg_rating": round(float(row.get("global_avg_rating", 0.0)), 2),
            }

        if "user_features" in ml_artifacts:
            ml_artifacts["uf_lookup"] = ml_artifacts["user_features"].set_index("userId")

        ml_artifacts["movie_feat_lookup"] = movies.set_index("movieId")[
            ["global_avg_rating", "rating_count_log"]
        ].to_dict("index")

        pop = movies.nlargest(200, "rating_count")[["movieId", "rating_count"]].copy()
        ml_artifacts["popularity_candidates"] = list(
            zip(pop["movieId"].values, pop["rating_count"].values.astype(float))
        )

    print("=" * 60 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_ml_artifacts()
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Movie Recommendation System",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _movie_payload(row: pd.Series) -> dict:
    genre_col = "genres_ml" if "genres_ml" in row.index else "genres_tmdb"
    title = row.get("title_ml", row.get("title_tmdb", "Unknown"))
    return {
        "movieId": int(row["movieId"]),
        "title": str(title),
        "genres": str(row.get(genre_col, "")),
        "global_avg_rating": round(float(row.get("global_avg_rating", 0.0)), 2),
    }


@app.get("/search")
async def search_movies(q: str = ""):
    """Search movies by title in the loaded ML catalog."""
    movies_df = ml_artifacts.get("movies_df")
    if movies_df is None:
        raise HTTPException(
            status_code=503,
            detail="ML artifacts not loaded. Server is still starting up.",
        )

    query = q.strip()
    if not query:
        return {"query": q, "results": []}

    title_col = "title_ml" if "title_ml" in movies_df.columns else "title_tmdb"
    matches = movies_df[
        movies_df[title_col].astype(str).str.contains(query, case=False, na=False, regex=False)
    ].head(30)

    return {
        "query": q,
        "results": [_movie_payload(row) for _, row in matches.iterrows()],
    }


def _score_and_rerank(user_id: int) -> list[dict]:
    """
    Full hybrid pipeline for a single user:
    1. Merge CF and content candidates.
    2. Score with the LightGBM ranker.
    3. Apply watched filtering, quality boosts, and diversity reranking.
    4. Return enriched top-10 recommendations.
    """
    cf_cands = ml_artifacts.get("cf_candidates", {})
    content_cands = ml_artifacts.get("content_candidates", {})
    ranker = ml_artifacts.get("ranker")
    feature_cols = ml_artifacts.get("feature_columns", [])
    uf_lookup = ml_artifacts.get("uf_lookup")
    movie_feat = ml_artifacts.get("movie_feat_lookup", {})
    movies_df = ml_artifacts.get("movies_df")
    rating_lookup = ml_artifacts.get("rating_lookup", {})
    genre_lookup = ml_artifacts.get("genre_lookup", {})
    movie_info = ml_artifacts.get("movie_info", {})

    movie_scores: dict = {}
    for mid, score in cf_cands.get(user_id, []):
        movie_scores[mid] = {"cf_score": score, "content_score": 0.0}
    for mid, score in content_cands.get(user_id, []):
        if mid in movie_scores:
            movie_scores[mid]["content_score"] = score
        else:
            movie_scores[mid] = {"cf_score": 0.0, "content_score": score}

    if not movie_scores or ranker is None:
        return []

    u_avg, u_count, u_entropy = 0.0, 0.0, 0.0
    if uf_lookup is not None and user_id in uf_lookup.index:
        uf = uf_lookup.loc[user_id]
        u_avg = float(uf["avg_rating_given"])
        u_count = float(uf["rating_count"])
        u_entropy = float(uf["genre_entropy"])

    mids = []
    rows = []
    for mid, scores in movie_scores.items():
        mf = movie_feat.get(mid, {})
        rows.append([
            scores.get("cf_score", 0.0),
            scores.get("content_score", 0.0),
            mf.get("global_avg_rating", 0.0),
            mf.get("rating_count_log", 0.0),
            u_avg,
            u_count,
            u_entropy,
        ])
        mids.append(mid)

    X = pd.DataFrame(rows, columns=feature_cols, dtype=np.float32)
    probs = ranker.predict_proba(X)[:, 1]
    order = np.argsort(-probs)
    scored_candidates = [(mids[i], float(probs[i])) for i in order]

    final_ids = rerank(
        scored_candidates,
        set(),
        movies_df,
        _rating_lookup=rating_lookup,
        _genre_lookup=genre_lookup,
    )

    score_map = {mid: sc for mid, sc in scored_candidates}
    results = []
    for mid in final_ids:
        info = movie_info.get(mid, {})
        results.append({
            "movieId": int(mid),
            "title": info.get("title", "Unknown"),
            "genres": info.get("genres", ""),
            "global_avg_rating": info.get("global_avg_rating", 0.0),
            "score": round(score_map.get(mid, 0.0), 4),
        })

    return results


def _popularity_fallback() -> list[dict]:
    """Return top-10 popular movies through the rerank filter."""
    pop_cands = ml_artifacts.get("popularity_candidates", [])
    movies_df = ml_artifacts.get("movies_df")
    rating_lookup = ml_artifacts.get("rating_lookup", {})
    genre_lookup = ml_artifacts.get("genre_lookup", {})
    movie_info = ml_artifacts.get("movie_info", {})

    final_ids = rerank(
        pop_cands,
        set(),
        movies_df,
        _rating_lookup=rating_lookup,
        _genre_lookup=genre_lookup,
    )

    score_map = {mid: sc for mid, sc in pop_cands}
    results = []
    for mid in final_ids:
        info = movie_info.get(mid, {})
        results.append({
            "movieId": int(mid),
            "title": info.get("title", "Unknown"),
            "genres": info.get("genres", ""),
            "global_avg_rating": info.get("global_avg_rating", 0.0),
            "score": round(score_map.get(mid, 0.0), 4),
        })
    return results


@app.get("/recommend/{user_id}")
async def recommend_for_user(user_id: int):
    """
    Personalized recommendations for a given user_id.

    Pipeline: merge CF and content candidates, score with LightGBM, rerank, and
    return the top 10. Falls back to popularity when the user is unknown.
    """
    if not ml_artifacts.get("ranker"):
        raise HTTPException(
            status_code=503,
            detail="ML artifacts not loaded. Server is still starting up.",
        )

    cf_cands = ml_artifacts.get("cf_candidates", {})
    content_cands = ml_artifacts.get("content_candidates", {})
    user_has_candidates = user_id in cf_cands or user_id in content_cands

    if user_has_candidates:
        recs = _score_and_rerank(user_id)
        if recs:
            return {"user_id": user_id, "source": "hybrid_lgbm", "recommendations": recs}

    recs = _popularity_fallback()
    return {"user_id": user_id, "source": "popularity_fallback", "recommendations": recs}


@app.get("/similar/{movie_title:path}")
async def similar_movies(movie_title: str):
    """
    Find the 10 most similar movies to the given title using TF-IDF and FAISS
    cosine similarity.
    """
    movies_df = ml_artifacts.get("movies_df")
    faiss_index = ml_artifacts.get("faiss_index")
    movie_id_to_idx = ml_artifacts.get("movie_id_to_idx")
    idx_to_movie_id = ml_artifacts.get("idx_to_movie_id")
    dense_normed = ml_artifacts.get("dense_normed")
    movie_info = ml_artifacts.get("movie_info", {})

    if movies_df is None or faiss_index is None:
        raise HTTPException(
            status_code=503,
            detail="ML artifacts not loaded. Server is still starting up.",
        )

    all_titles = movies_df["title_ml"].astype(str).tolist()
    matches = difflib.get_close_matches(movie_title, all_titles, n=1, cutoff=0.4)

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No movie found matching '{movie_title}'.",
        )

    matched_title = matches[0]
    matched_row = movies_df[movies_df["title_ml"] == matched_title].iloc[0]
    matched_mid = int(matched_row["movieId"])

    if matched_mid not in movie_id_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Movie '{matched_title}' not indexed for similarity.",
        )

    query_idx = movie_id_to_idx[matched_mid]
    query_vec = dense_normed[query_idx].reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, 11)

    results = []
    for j in range(11):
        idx = int(indices[0, j])
        if idx < 0:
            continue

        mid = idx_to_movie_id.get(idx)
        if mid is None or mid == matched_mid:
            continue

        info = movie_info.get(mid, {})
        results.append({
            "movieId": int(mid),
            "title": info.get("title", "Unknown"),
            "genres": info.get("genres", ""),
            "global_avg_rating": info.get("global_avg_rating", 0.0),
            "similarity_score": round(float(scores[0, j]), 4),
        })

        if len(results) >= 10:
            break

    return {
        "query": movie_title,
        "matched_title": matched_title,
        "similar_movies": results,
    }
