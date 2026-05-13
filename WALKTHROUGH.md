# Hybrid Movie Recommender — Project Walkthrough

**Author:** Aryan Singh  
**College:** Panipat Institute of Engineering and Technology  
**Branch:** B.Tech CSE (AI & DS)

---

## What This Project Is

A production-style movie recommendation system built in phases. It goes beyond a simple "similar movies" finder — it personalizes recommendations per user based on their rating history, using a hybrid pipeline with collaborative filtering, content retrieval, and a learned ranking model.

**Starting point:** A working FastAPI app with TF-IDF cosine similarity, SQLite auth, and a watchlist.  
**End goal:** A two-stage recommender — candidate generation → ranking → re-ranking — served via API.

---

## Dataset Overview

| Dataset | Source | Size |
|---|---|---|
| MovieLens 25M | grouplens.org/datasets/movielens/25m | 25M ratings, 162K users, 62K movies |
| TMDB metadata | Kaggle (TMDB dataset) | ~1M rows, movie metadata |
| Merged output | After ingest.py | 9,377 movies, 20.2M ratings, 152K active users |

The two datasets are joined on `tmdbId` (primary) and `clean_title` (fallback). 9,377 matched movies cover 81% of all ratings — solid coverage.

---

## Project Structure

```
Movie_Recommendation/
│
├── data/
│   ├── raw/
│   │   ├── ml-25m/              ← MovieLens 25M (downloaded)
│   │   │   ├── ratings.csv
│   │   │   ├── movies.csv
│   │   │   └── links.csv
│   │   └── TMDB_all_movies.csv  ← TMDB metadata (original)
│   │
│   ├── processed/               ← Output of pipeline scripts
│   │   ├── movies_merged.csv    ← ML + TMDB joined (9,377 × 37)
│   │   ├── ratings_filtered.csv ← 20.2M ratings, 152K users
│   │   ├── movies_final.csv     ← + tags + item stats
│   │   ├── train.csv            ← 80% per-user interactions
│   │   └── test.csv             ← 20% most-recent per user
│   │
│   └── artifacts/               ← Saved model files
│       ├── cf_model.pkl         ← Trained SVD model
│       ├── faiss_index.bin      ← ANN vector index
│       ├── tfidf_matrix.pkl     ← TF-IDF vectors
│       ├── lgbm_ranker.pkl      ← LightGBM ranking model
│       └── encoders.pkl         ← Label encoders, scalers
│
├── src/
│   ├── data/
│   │   ├── ingest.py            ← Merge MovieLens + TMDB
│   │   └── preprocess.py        ← Tags, train/test split, item stats
│   │
│   ├── candidates/
│   │   ├── cf_candidates.py     ← SVD → top-200 CF candidates per user
│   │   └── content_candidates.py← FAISS → top-200 content candidates per user
│   │
│   ├── features/
│   │   ├── user_features.py     ← Genre affinity, avg rating, activity
│   │   ├── item_features.py     ← Popularity, global rating, recency
│   │   └── interaction_features.py ← (user, movie) pair features for ranker
│   │
│   ├── ranker/
│   │   ├── feature_builder.py   ← Build feature matrix for all candidates
│   │   ├── train_ranker.py      ← Train LightGBM ranking model
│   │   └── rerank.py            ← Diversity, dedup, quality filter
│   │
│   └── evaluation/
│       ├── metrics.py           ← Precision@K, Recall@K, NDCG@K, Coverage
│       └── evaluate.py          ← Run all baselines + final model, print table
│
├── app/                         ← Existing FastAPI app (keep as-is)
│   ├── main.py                  ← Add /recommend/{user_id} route here
│   ├── recommender.py           ← Extend to use hybrid pipeline
│   ├── models.py
│   ├── database.py
│   └── templates/
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Explore merged dataset
│   ├── 02_cf_experiment.ipynb   ← SVD hyperparameter tuning
│   └── 03_ranker_experiment.ipynb
│
├── pipeline.py                  ← Run full offline training end-to-end
├── clean_data.py                ← Original cleaning script (keep)
├── TMDB_all_movies.csv          ← Original TMDB file (keep)
├── requirements.txt
├── README.md
└── WALKTHROUGH.md               ← This file
```

---

## System Architecture

```
User request: /recommend/{user_id}
        │
        ▼
┌─────────────────────────────┐
│   Stage A: Candidate Gen    │
│                             │
│  CF candidates (SVD)        │  ← 200 movies based on similar users
│       +                     │
│  Content candidates (FAISS) │  ← 200 movies based on watch history
│       ↓                     │
│  Union → ~300-400 candidates│
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage B: Ranking          │
│                             │
│  Build (user, movie) feats  │
│  Score via LightGBM ranker  │
│  Sort by predicted score    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage C: Re-ranking       │
│                             │
│  Remove already-watched     │
│  Cap same genre > 3         │
│  Min rating threshold ≥ 4.0 │
│  Return top-10              │
└────────────┬────────────────┘
             │
             ▼
        JSON response
```

---

## Build Phases

### Phase 1 — Data + Baselines
**Goal:** Get data ready, establish what we're improving over.

| Step | Script | Status |
|---|---|---|
| Download MovieLens 25M | manual | ✅ Done |
| Merge MovieLens + TMDB | `src/data/ingest.py` | ✅ Done |
| Train/test split + tags | `src/data/preprocess.py` | ✅ Done |
| Train SVD (CF baseline) | `src/candidates/cf_candidates.py` | ⬜ Next |
| Build TF-IDF content index | `src/candidates/content_candidates.py` | ⬜ |
| Popularity baseline | `src/evaluation/evaluate.py` | ⬜ |

### Phase 2 — Hybrid Pipeline + Ranker
**Goal:** Beat baselines with a learned ranking model.

| Step | Script | Status |
|---|---|---|
| User feature engineering | `src/features/user_features.py` | ⬜ |
| Item feature engineering | `src/features/item_features.py` | ⬜ |
| Hybrid candidate merge | `src/ranker/feature_builder.py` | ⬜ |
| Train LightGBM ranker | `src/ranker/train_ranker.py` | ⬜ |
| Re-ranking logic | `src/ranker/rerank.py` | ⬜ |
| Evaluation vs baselines | `src/evaluation/evaluate.py` | ⬜ |

### Phase 3 — Serving + UI
**Goal:** Expose the pipeline via API, show results in the existing UI.

| Step | Script | Status |
|---|---|---|
| Add /recommend/{user_id} endpoint | `app/main.py` | ⬜ |
| Load artifacts at startup | `app/recommender.py` | ⬜ |
| Add recommendations panel to homepage | `app/templates/` | ⬜ |
| Write full README | `README.md` | ⬜ |

---

## Key Numbers (after ingest.py)

| Metric | Value |
|---|---|
| Matched movies | 9,377 |
| Total filtered ratings | 20,209,535 |
| Active users (≥ 20 ratings) | 152,160 |
| Rating coverage | 81% of all 25M ratings |
| Match via tmdbId | 7,726 movies |
| Match via title fallback | 1,651 movies |

---

## Evaluation Plan

We compare 4 models. The hybrid must beat all baselines — this is the core claim.

| Model | Description | Expected NDCG@10 |
|---|---|---|
| Popularity baseline | Recommend globally most-rated movies | ~0.05 |
| Content-only | TF-IDF cosine (current system) | ~0.07 |
| CF-only | SVD collaborative filtering | ~0.14 |
| **Hybrid + Ranker** | CF + content + LightGBM | **~0.20+** |

Split: time-aware per user (last 20% interactions = test).  
Metrics: Precision@10, Recall@10, NDCG@10, Coverage, Diversity.

---

## Features Used in Ranker

**User features**
- avg_rating_given
- total_ratings_count
- top_3_preferred_genres
- genre_entropy (how diverse is their taste)
- decade_preference (do they like old or new movies)

**Item features**
- genres (multi-hot encoded)
- global_avg_rating
- rating_count_log
- tmdb_popularity
- release_year

**Interaction features (user × movie)**
- cf_predicted_score
- content_similarity_score
- genre_overlap_score
- user_avg_rating_for_this_genre
- popularity_percentile

---

## How to Run (end to end)

```bash
# Step 1 — Merge datasets
python src/data/ingest.py

# Step 2 — Preprocess
python src/data/preprocess.py

# Step 3 — Train CF model
python src/candidates/cf_candidates.py

# Step 4 — Build content index
python src/candidates/content_candidates.py

# Step 5 — Train ranker
python src/ranker/train_ranker.py

# Step 6 — Evaluate all models
python src/evaluation/evaluate.py

# Step 7 — Start API
uvicorn app.main:app --reload
```

Or run everything at once:
```bash
python pipeline.py
```

---

## What's Different from the Original Project

| Feature | Before (v1) | After (v2) |
|---|---|---|
| Recommendation type | Same for everyone | Personalized per user |
| Algorithm | TF-IDF cosine only | CF + Content + LightGBM ranker |
| User data used | None | 20M ratings from MovieLens |
| Evaluation | Manual / none | Precision@K, NDCG@K vs baselines |
| Candidate pool | All movies, cosine ranked | 300-400 per user, then ranked |
| API | /search only | /recommend/{user_id} + /similar/{title} |

---

