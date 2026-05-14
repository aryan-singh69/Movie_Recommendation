"""
metrics.py — Recommendation-system evaluation metrics.

Implements:
  - Precision@K
  - Recall@K
  - NDCG@K
  - Coverage  (catalog coverage across all users)
  - Diversity (avg intra-list genre diversity per user)

All per-user metrics are macro-averaged across users.
"""

import numpy as np


# ══════════════════════════════════════════════
#  Precision@K
# ══════════════════════════════════════════════
def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Fraction of the top-K recommended items that are relevant.

    Parameters
    ----------
    recommended : list of int
        Ordered list of movieIds (most relevant first).
    relevant : set of int
        Set of movieIds the user rated ≥ 4.0 in the test set.
    k : int
        Cut-off.

    Returns
    -------
    float
    """
    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / len(top_k)


# ══════════════════════════════════════════════
#  Recall@K
# ══════════════════════════════════════════════
def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Fraction of the user's relevant items that appear in the top-K.

    Parameters
    ----------
    recommended : list of int
    relevant : set of int
    k : int

    Returns
    -------
    float
    """
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / len(relevant)


# ══════════════════════════════════════════════
#  NDCG@K
# ══════════════════════════════════════════════
def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    Uses binary relevance: gain = 1 if item is relevant, else 0.

    Parameters
    ----------
    recommended : list of int
    relevant : set of int
    k : int

    Returns
    -------
    float
    """
    top_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, mid in enumerate(top_k):
        if mid in relevant:
            dcg += 1.0 / np.log2(i + 2)  # position 0 → log2(2)

    # Ideal DCG — place all relevant items at the top positions
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ══════════════════════════════════════════════
#  Coverage
# ══════════════════════════════════════════════
def coverage(all_recommendations: dict, total_movies: int) -> float:
    """
    Fraction of the total movie catalog that appears in at least one
    user's recommendation list.

    Parameters
    ----------
    all_recommendations : dict
        {userId: [movieId, ...]}
    total_movies : int
        Size of the full catalog.

    Returns
    -------
    float
    """
    if total_movies == 0:
        return 0.0
    unique_movies = set()
    for recs in all_recommendations.values():
        unique_movies.update(recs)
    return len(unique_movies) / total_movies


# ══════════════════════════════════════════════
#  Diversity (intra-list genre diversity)
# ══════════════════════════════════════════════
def intra_list_diversity(recommended: list, genre_lookup: dict) -> float:
    """
    Intra-list genre diversity for a single user.

    Defined as  1 − genre_concentration, where genre_concentration is the
    Herfindahl index (sum of squared genre-share fractions) computed over
    all genres that appear in the recommendation list.

    A perfectly diverse list (every item a different genre) → diversity ≈ 1.
    A perfectly homogeneous list (all same genre) → diversity = 0.

    Parameters
    ----------
    recommended : list of int
        movieIds in the recommendation list.
    genre_lookup : dict
        movieId → [genre1, genre2, ...].

    Returns
    -------
    float
    """
    if len(recommended) == 0:
        return 0.0

    genre_counts: dict[str, int] = {}
    total = 0
    for mid in recommended:
        genres = genre_lookup.get(mid, [])
        primary = genres[0] if genres else "Unknown"
        genre_counts[primary] = genre_counts.get(primary, 0) + 1
        total += 1

    if total == 0:
        return 0.0

    # Herfindahl index (concentration)
    concentration = sum((c / total) ** 2 for c in genre_counts.values())
    return 1.0 - concentration
