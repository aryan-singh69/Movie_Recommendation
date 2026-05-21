from contextlib import asynccontextmanager
import difflib
import hashlib
import os
import pickle
import sys
import urllib.parse

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.sparse import issparse
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from . import models, schemas, database, recommender
from .models import User, Watchlist

from .database import engine
from .models import Base

Base.metadata.create_all(bind=engine)

# ──────────────────────────────────────────────
# Path configuration for ML artifacts
# ──────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "data", "artifacts")

# Allow imports from project root (for rerank)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.ranker.rerank import rerank, _build_lookups, _parse_genres

# ──────────────────────────────────────────────
# Global store for loaded ML artifacts
# ──────────────────────────────────────────────
ml_artifacts: dict = {}


def _load_pkl(name: str):
    path = os.path.join(_ARTIFACTS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ml_artifacts():
    """Load all ML artifacts into the global store. Called once at startup."""
    print("\n" + "=" * 60)
    print("  STARTUP — Loading ML artifacts")
    print("=" * 60)

    # --- Candidate dicts ---
    try:
        ml_artifacts["cf_candidates"] = _load_pkl("cf_candidates.pkl")
        print(f"  ✓ cf_candidates.pkl          ({len(ml_artifacts['cf_candidates']):,} users)")
    except Exception as e:
        print(f"  ✗ cf_candidates.pkl          FAILED: {e}")

    try:
        ml_artifacts["content_candidates"] = _load_pkl("content_candidates.pkl")
        print(f"  ✓ content_candidates.pkl     ({len(ml_artifacts['content_candidates']):,} users)")
    except Exception as e:
        print(f"  ✗ content_candidates.pkl     FAILED: {e}")

    # --- Ranker ---
    try:
        ml_artifacts["ranker"] = _load_pkl("lgbm_ranker.pkl")
        print(f"  ✓ lgbm_ranker.pkl            loaded")
    except Exception as e:
        print(f"  ✗ lgbm_ranker.pkl            FAILED: {e}")

    try:
        ml_artifacts["user_features"] = _load_pkl("user_features.pkl")
        print(f"  ✓ user_features.pkl          ({ml_artifacts['user_features'].shape})")
    except Exception as e:
        print(f"  ✗ user_features.pkl          FAILED: {e}")

    try:
        ml_artifacts["feature_columns"] = _load_pkl("feature_columns.pkl")
        print(f"  ✓ feature_columns.pkl        {ml_artifacts['feature_columns']}")
    except Exception as e:
        print(f"  ✗ feature_columns.pkl        FAILED: {e}")

    # --- Movies metadata ---
    try:
        movies_path = os.path.join(_PROCESSED_DIR, "movies_final.csv")
        ml_artifacts["movies_df"] = pd.read_csv(movies_path, encoding="latin-1")
        print(f"  ✓ movies_final.csv           ({ml_artifacts['movies_df'].shape})")
    except Exception as e:
        print(f"  ✗ movies_final.csv           FAILED: {e}")

    # --- movie_id_to_idx (for FAISS similar-movies) ---
    try:
        ml_artifacts["movie_id_to_idx"] = _load_pkl("movie_id_to_idx.pkl")
        ml_artifacts["idx_to_movie_id"] = _load_pkl("idx_to_movie_id.pkl")
        print(f"  ✓ movie_id_to_idx.pkl        ({len(ml_artifacts['movie_id_to_idx']):,} movies)")
    except Exception as e:
        print(f"  ✗ movie_id_to_idx.pkl        FAILED: {e}")

    # --- TF-IDF matrix + FAISS index (rebuild from matrix) ---
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
        print(f"  ✓ tfidf_matrix.pkl           shape={mat.shape}")
        print(f"  ✓ FAISS index rebuilt         ({index.ntotal:,} vectors)")
    except Exception as e:
        print(f"  ✗ tfidf_matrix / FAISS       FAILED: {e}")

    # --- Pre-build rerank lookups ---
    if "movies_df" in ml_artifacts:
        movies = ml_artifacts["movies_df"]
        rating_lk, genre_lk = _build_lookups(movies)
        ml_artifacts["rating_lookup"] = rating_lk
        ml_artifacts["genre_lookup"] = genre_lk

        # Movie info lookup for JSON responses
        ml_artifacts["movie_info"] = {}
        genre_col = "genres_ml" if "genres_ml" in movies.columns else "genres_tmdb"
        for _, row in movies.iterrows():
            ml_artifacts["movie_info"][row["movieId"]] = {
                "movieId": int(row["movieId"]),
                "title": str(row.get("title_ml", row.get("title_tmdb", "Unknown"))),
                "genres": str(row.get(genre_col, "")),
                "global_avg_rating": round(float(row.get("global_avg_rating", 0.0)), 2),
            }

        # User-feature lookup indexed by userId
        if "user_features" in ml_artifacts:
            ml_artifacts["uf_lookup"] = ml_artifacts["user_features"].set_index("userId")

        # Movie-feature lookup for ranker scoring
        ml_artifacts["movie_feat_lookup"] = movies.set_index("movieId")[
            ["global_avg_rating", "rating_count_log"]
        ].to_dict("index")

        # Popularity fallback: top-200 movies by rating count in the catalog
        pop = movies.nlargest(200, "rating_count")[["movieId", "rating_count"]].copy()
        ml_artifacts["popularity_candidates"] = list(
            zip(pop["movieId"].values, pop["rating_count"].values.astype(float))
        )

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# Lifespan: load artifacts at startup
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_ml_artifacts()
    yield
    ml_artifacts.clear()


app = FastAPI(title="Movie Recommendation System", lifespan=lifespan)

# Integrate simple, secure session-based authentication without overcomplicating with JWT!
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_123")

os.makedirs("app/static", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    return hash_password(password) == hashed_password

def get_current_user_id(request: Request):
    return request.session.get("user_id")

from .database import get_db

# ─── CORE USER & VIEW ROUTES ─────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    movies = recommender.get_movies()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "movies": movies,
        "current_user": get_current_user_id(request)
    })

@app.get("/search", response_class=HTMLResponse)
async def search_movies(request: Request, q: str = ""):
    results = recommender.search_movies(q) if q else []
    return templates.TemplateResponse("search.html", {
        "request": request, 
        "query": q, 
        "movies": results,
        "current_user": get_current_user_id(request)
    })

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, movie_title: str = Form(...)):
    recommendations = recommender.recommend_movies(movie_title, top_n=10)
    
    if not recommendations:
        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": movie_title,
            "movies": [],
            "error": f"Movie titled '{movie_title}' was not found in our datasets or no recommendations exist.",
            "current_user": get_current_user_id(request)
        })
        
    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "query": movie_title,
        "recommendations": recommendations,
        "current_user": get_current_user_id(request)
    })

@app.get("/movie/{movie_title}", response_class=HTMLResponse)
async def get_movie_details(request: Request, movie_title: str, db: Session = Depends(get_db)):
    results = recommender.search_movies(movie_title)
    
    exact_match = None
    for m in results:
        if str(m['title']).lower() == movie_title.lower():
            exact_match = m
            break
            
    if not exact_match and results:
        exact_match = results[0] 
        
    if not exact_match:
        raise HTTPException(status_code=404, detail=f"Movie details for '{movie_title}' could not be located.")
        
    # Natively query SQL for the mathematical average of specific Community Ratings!
    from sqlalchemy.sql import func
    avg_rating_query = db.query(func.avg(models.Rating.rating)).filter(models.Rating.movie_title == exact_match["title"]).scalar()
    avg_user_rating = round(avg_rating_query, 1) if avg_rating_query else None
    
    # Pre-source the user's explicit rating history to map back into UI natively 
    current_user_id = get_current_user_id(request)
    user_rating_val = None
    if current_user_id:
        my_rating = db.query(models.Rating).filter(
            models.Rating.user_id == current_user_id, 
            models.Rating.movie_title == exact_match["title"]
        ).first()
        if my_rating:
            user_rating_val = my_rating.rating

    recommendations = recommender.recommend_movies(exact_match["title"], top_n=5)
    
    return templates.TemplateResponse("movie_detail.html", {
        "request": request, 
        "movie": exact_match, 
        "recommendations": recommendations,
        "current_user": current_user_id,
        "avg_user_rating": avg_user_rating,
        "user_rating": user_rating_val
    })

@app.post("/rate")
async def rate_movie(request: Request, movie_title: str = Form(...), rating: int = Form(...), db: Session = Depends(get_db)):
    current_user_id = get_current_user_id(request)
    if not current_user_id:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        
    rating = max(1, min(5, rating)) # Safeguard between 1 and 5
        
    # Intelligently update existing scores preventing db clashes 
    existing_rating = db.query(models.Rating).filter(
        models.Rating.user_id == current_user_id,
        models.Rating.movie_title == movie_title
    ).first()
    
    if existing_rating:
        existing_rating.rating = rating
    else:
        new_rating = models.Rating(user_id=current_user_id, movie_title=movie_title, rating=rating)
        db.add(new_rating)
        
    db.commit()
    
    # Loop user reliably back using safe encoded URL structures 
    encoded_title = urllib.parse.quote(movie_title.strip())
    return RedirectResponse(url=f"/movie/{encoded_title}", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/popular", response_class=HTMLResponse)
async def popular_page(request: Request):
    movies = recommender.get_popular_movies(top_n=30, min_votes=100)
    return templates.TemplateResponse("popular.html", {
        "request": request, 
        "movies": movies,
        "current_user": get_current_user_id(request)
    })

# ─── SECURE AUTHENTICATION SYSTEM ────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password"
        })

    request.session["user_id"] = user.id
    request.session["username"] = user.username

    return RedirectResponse(url="/", status_code=303)

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})

@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()

    if existing_user:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Username or email already exists"
        })

    new_user = User(
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=303)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)

# ─── SECURE WATCHLIST WITH SESSIONS ──────────────────────

@app.post("/watchlist/add")
async def add_to_watchlist(
    request: Request,
    movie_title: str = Form(...),
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    existing = db.query(Watchlist).filter(
        Watchlist.user_id == user_id,
        Watchlist.movie_title == movie_title
    ).first()

    if not existing:
        item = Watchlist(user_id=user_id, movie_title=movie_title)
        db.add(item)
        db.commit()

    return RedirectResponse(url="/watchlist", status_code=303)


@app.get("/watchlist", response_class=HTMLResponse)
async def view_watchlist(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()

    return templates.TemplateResponse("watchlist.html", {
        "request": request,
        "watchlist": items,
        "username": request.session.get("username")
    })


@app.post("/watchlist/remove")
async def remove_from_watchlist(
    request: Request,
    movie_title: str = Form(...),
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    item = db.query(Watchlist).filter(
        Watchlist.user_id == user_id,
        Watchlist.movie_title == movie_title
    ).first()

    if item:
        db.delete(item)
        db.commit()

    return RedirectResponse(url="/watchlist", status_code=303)


# ─── ML-POWERED RECOMMENDATION API ──────────────────────

def _score_and_rerank(user_id: int) -> list[dict]:
    """
    Full hybrid pipeline for a single user:
      1. Merge CF + content candidates
      2. Score with LightGBM ranker
      3. Apply rerank (watched filter, quality, diversity)
      4. Return enriched top-10 dicts
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

    # --- Merge CF + content candidates ---
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

    # --- User features ---
    u_avg, u_count, u_entropy = 0.0, 0.0, 0.0
    if uf_lookup is not None and user_id in uf_lookup.index:
        uf = uf_lookup.loc[user_id]
        u_avg = float(uf["avg_rating_given"])
        u_count = float(uf["rating_count"])
        u_entropy = float(uf["genre_entropy"])

    # --- Build feature matrix & score ---
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

    # --- Rerank ---
    watched = set()  # rerank itself handles watched filtering via train data
    final_ids = rerank(
        scored_candidates, watched, movies_df,
        _rating_lookup=rating_lookup, _genre_lookup=genre_lookup,
    )

    # --- Enrich with metadata ---
    results = []
    # Map movieId → score for the scored list
    score_map = {mid: sc for mid, sc in scored_candidates}
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
        pop_cands, set(), movies_df,
        _rating_lookup=rating_lookup, _genre_lookup=genre_lookup,
    )

    results = []
    score_map = {mid: sc for mid, sc in pop_cands}
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


@app.get("/recommend/popular")
async def recommend_popular():
    """Return top-10 popular movies (non-personalized fallback)."""
    if not ml_artifacts.get("popularity_candidates"):
        raise HTTPException(
            status_code=503,
            detail="ML artifacts not loaded. Server is still starting up.",
        )
    recs = _popularity_fallback()
    return {"user_id": None, "source": "popularity_fallback", "recommendations": recs}


@app.get("/recommend/{user_id}")
async def recommend_for_user(user_id: int):
    """
    Personalized recommendations for a given user_id.

    Pipeline: merge CF + content candidates → LightGBM scoring → rerank → top 10.
    Falls back to popularity-based recommendations if user not found.
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

    # Fallback: popularity
    recs = _popularity_fallback()
    return {"user_id": user_id, "source": "popularity_fallback", "recommendations": recs}


# ─── SIMILAR MOVIES API ─────────────────────────────────

@app.get("/similar/{movie_title:path}")
async def similar_movies(movie_title: str):
    """
    Find the 10 most similar movies to the given title using
    TF-IDF + FAISS cosine similarity.

    Uses difflib fuzzy matching to find the closest title in the catalog.
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

    # --- Fuzzy match title ---
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

    # --- Query FAISS ---
    query_idx = movie_id_to_idx[matched_mid]
    query_vec = dense_normed[query_idx].reshape(1, -1)
    scores, indices = faiss_index.search(query_vec, 11)  # +1 for self

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
