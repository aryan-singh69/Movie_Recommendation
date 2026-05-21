import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Create an absolute path to load the CSV irrespective of where FastAPI is launched from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_movies.csv")

# Initialize global variables for FastAPI state
df_movies = pd.DataFrame()
tfidf_matrix = None
tfidf = TfidfVectorizer(stop_words='english')

try:
    print(f"Loading datasets and computing TF-IDF Matrix from '{CSV_PATH}'...")
    df_movies = pd.read_csv(CSV_PATH, low_memory=False)
    
    # Apply exact global mathematical precision recursively avoiding messy Jinja overrides
    if 'vote_average' in df_movies.columns:
        df_movies['vote_average'] = df_movies['vote_average'].round(1)

    # Create a lowercase title column for robust, case-insensitive searching
    if 'title' in df_movies.columns:
        df_movies['title_lower'] = df_movies['title'].astype(str).str.lower()
except Exception as e:
    print(f"Warning: Failed to load dataset. Make sure 'cleaned_movies.csv' exists. Details: {e}")

# Precompute the TF-IDF matrix once on startup to maintain highly responsive request times
if not df_movies.empty and 'tags' in df_movies.columns:
    df_movies['tags'] = df_movies['tags'].fillna('')
    try:
        tfidf_matrix = tfidf.fit_transform(df_movies['tags'])
        print("TF-IDF Matrix successfully computed!")
    except Exception as e:
        print(f"Warning: Failed to build TF-IDF matrix. Details: {e}")


def get_popular_movies(top_n: int = 10, min_votes: int = 50):
    """
    Returns popular movies utilizing a combination of vote_average and vote_count.
    Ensures a minimum number of votes (min_votes) for statistical reliability.
    """
    if df_movies.empty:
        return []

    # Verify if necessary columns exist
    required_cols = ['vote_count', 'vote_average', 'popularity']
    # Fallback if any required column is missing
    if not all(c in df_movies.columns for c in required_cols):
        return df_movies.head(top_n).to_dict(orient='records')
    
    # Filter out movies with low vote counts to avoid outliers 
    # (e.g. a movie with 1 vote of 10.0 being ranked #1)
    filtered_df = df_movies[df_movies['vote_count'] >= min_votes]
    
    if filtered_df.empty:
        # Fallback if the selected threshold is too strict for the given dataset
        filtered_df = df_movies 
        
    sorted_df = filtered_df.sort_values(by=['vote_average', 'popularity'], ascending=[False, False])
    return sorted_df.head(top_n).to_dict(orient='records')


def recommend_movies(movie_title: str, top_n: int = 10):
    """
    Given a movie title (case-insensitive), returns a list of recommended movies
    based on TF-IDF cosine similarity against the unified 'tags' column.
    """
    # 1. Handle cases where dataframe or matrix is empty
    if df_movies.empty or tfidf_matrix is None:
        return []
        
    # 2. Make the requested search case-insensitive
    movie_title_lower = str(movie_title).lower().strip()
    
    # 3. Handle cases where the movie title is completely not found
    if movie_title_lower not in df_movies['title_lower'].values:
        print(f"Title '{movie_title}' not found in the dataset.")
        return []
        
    # Locate the internal index of the requested movie
    idx = df_movies[df_movies['title_lower'] == movie_title_lower].index[0]
    
    # Extract the TF-IDF vector specifically for the queried movie
    target_vector = tfidf_matrix[idx]
    
    # 4. Compute mathematical cosine similarity strictly between the target movie and all other movies
    # This design prevents Out-Of-Memory errors compared to computing/storing the massive N x N matrix
    sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
    
    # Enumerate and sort the movies systematically based on similarity score (excluding exact matches)
    sim_scores_indexed = list(enumerate(sim_scores))
    sim_scores_sorted = sorted(sim_scores_indexed, key=lambda x: x[1], reverse=True)
    
    # Drop the queried movie itself (which will always be index 0 after sorting, similarity 1.0)
    sim_scores_sorted = sim_scores_sorted[1:top_n+1]
    
    # Gather the indices of the strictly recommended movies
    movie_indices = [i[0] for i in sim_scores_sorted]
    
    # 5. Filter for the specifically requested return columns
    result_df = df_movies.iloc[movie_indices]
    cols_to_return = ['id', 'title', 'genres', 'vote_average', 'popularity', 'poster_path']
    existing_cols = [col for col in cols_to_return if col in result_df.columns]
    
    return result_df[existing_cols].to_dict(orient='records')


# --- Adapter & Compatibility methods below for existing routing interfaces in main.py ---

def get_movies():
    """Wrapper logic to render foundational data for the home page."""
    # We populate the homepage directly with top_n popular movies
    return get_popular_movies(top_n=20, min_votes=100)

def search_movies(query: str):
    """Robust fallback search for generalized query strings."""
    if df_movies.empty: return []
    res = df_movies[df_movies['title'].str.contains(query, case=False, na=False)]
    return res.head(30).to_dict(orient='records')

def get_movie_by_id(movie_id: int):
    """Retrieve detailed specs for a specific movie directly by ID."""
    if df_movies.empty: return None
    movie = df_movies[df_movies['id'] == movie_id]
    if not movie.empty:
        return movie.iloc[0].to_dict()
    return None

def get_recommendations(title: str, num_recommendations: int = 3):
    """Compatability wrapper pointing to your newly created recommend_movies system."""
    return recommend_movies(movie_title=title, top_n=num_recommendations)
