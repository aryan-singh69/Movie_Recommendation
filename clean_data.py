import pandas as pd

def clean_movie_data(input_file: str, output_file: str):
    """
    Cleans movie dataset by handling missing values, encoding issues,
    and generating a unified tags column for recommendations.
    """
    print(f"Loading data from '{input_file}'...")
    
    # 1 & 2. Load the CSV safely, handling encoding and skipping bad lines
    # Using low_memory=False to prevent dtype warnings on large, slightly messy files
    try:
        df = pd.read_csv(
            input_file, 
            encoding='utf-8', 
            encoding_errors='replace', 
            on_bad_lines='skip',
            low_memory=False
        )
    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}'.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    initial_rows = len(df)
    print(f"Initial rows before cleaning: {initial_rows}")

    # 3. Remove fully blank rows (where all columns are NaN)
    df.dropna(how='all', inplace=True)

    # 4. Keep only useful columns
    useful_cols = [
        'id', 'title', 'genres', 'overview', 'cast', 'director', 
        'poster_path', 'vote_average', 'vote_count', 'release_date', 'popularity'
    ]
    
    # Filter to keep only columns that actually exist in the dataframe
    cols_to_keep = [col for col in useful_cols if col in df.columns]
    df = df[cols_to_keep]

    # 5. Drop rows where 'title' is missing
    if 'title' in df.columns:
        # Drop rows where title is NaN or an empty string
        df.dropna(subset=['title'], inplace=True)
        # If there are rows where title is just whitespace, treat them as missing and drop
        df = df[df['title'].astype(str).str.strip() != '']

    # 6. Fill missing overview, genres, cast, and director with empty strings
    text_cols = ['genres', 'overview', 'cast', 'director']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 7. Create a new combined text column called 'tags' using genres + overview + cast + director
    available_text_cols = [col for col in text_cols if col in df.columns]
    
    if available_text_cols:
        # Join the text columns together, separated by spaces
        df['tags'] = df[available_text_cols].astype(str).agg(' '.join, axis=1)
        # Clean up any multiple spaces and strip leading/trailing whitespace
        df['tags'] = df['tags'].str.replace(r'\s+', ' ', regex=True).str.strip()
    else:
        df['tags'] = ""

    # Robust cleanup: drop duplicate IDs if they exist and ID column is present
    if 'id' in df.columns:
        df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    df.reset_index(drop=True, inplace=True)

    # 9. Print valid rows before and after cleaning
    final_rows = len(df)
    print(f"Rows after cleaning: {final_rows}")
    print(f"Removed {initial_rows - final_rows} invalid or blank rows.")

    # 8. Save the cleaned data as cleaned_movies.csv
    print(f"Saving cleaned dataset to '{output_file}'...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print("Done! Data cleaning is complete.")

if __name__ == "__main__":
    INPUT_CSV = "TMDB_all_movies.csv"
    OUTPUT_CSV = "cleaned_movies.csv"
    
    clean_movie_data(INPUT_CSV, OUTPUT_CSV)
