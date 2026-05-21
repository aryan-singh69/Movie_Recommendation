# Movie Recommendation System 🎬

## 📌 Objective
The primary objective of this project is to provide a clean FastAPI backend for personalized movie recommendations. The current app focuses on ML-powered recommendation, similarity, and search endpoints, leaving authentication ready for a fresh implementation later.

## ✨ Features
* **Hybrid Recommendation API**: Combines collaborative and content candidates, scores them with a ranking model, then reranks for quality and diversity.
* **Similar Movies API**: Uses TF-IDF vectors and FAISS similarity search to find nearby titles.
* **Search API**: Searches the loaded movie catalog by title.

## 🛠️ Technology Stack
* **Backend Framework**: Python (FastAPI)
* **Machine Learning**: Pandas, NumPy, Scikit-Learn, FAISS, LightGBM ranking artifacts

## 📂 Dataset Used
This platform actively integrates a custom-trimmed slice of the **TMDB (The Movie Database)** dataset (`TMDB_all_movies.csv`). The CSV undergoes strict data sanitation using the project's native `clean_data.py` pipeline to drop null fields, patch encoding errors, and concatenate "Tags" needed for the AI engine prior to deployment.

## 🏗️ Project Architecture
The application is now an API-first FastAPI service.
* **API (`app/main.py`)**: Loads ML artifacts and exposes recommendation, similarity, and search endpoints.
* **Recommendation helpers (`src/ranker`)**: Applies reranking rules to model-scored candidates.
* **Data pipelines (`clean_data.py`, `src`)**: Prepare datasets and model artifacts used at startup.

## 🧩 Core Modules
1. **`clean_data.py`** – Parses raw large CSV constraints securely generating strict structural integrity.
2. **`app/main.py`** - Exposes the clean ML recommendation API.
3. **`src/ranker/rerank.py`** - Applies quality and diversity reranking to candidate lists.

---

## 🚀 Installation Steps
1. **Clone or Extract** the project repository locally onto your machine.
2. Ensure you have **Python 3.9+** actively installed.
3. Establish your terminal directory strictly at the folder root (`/Movie_Recommendation`).
4. Install all native requirements accurately:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ How to Run
Initialize the FastAPI server securely via the `uvicorn` production wrapper engine:
```bash
uvicorn app.main:app --reload
```
Once the log reads that startup is complete, call the API endpoints directly:
* `http://127.0.0.1:8000/search?q=toy`
* `http://127.0.0.1:8000/recommend/1`
* `http://127.0.0.1:8000/similar/Toy%20Story`

---

## 🔮 Future Scope
* **Fresh Authentication**: Add a new auth implementation on top of the clean API.
* **Frontend Client**: Build a separate web UI that consumes the FastAPI endpoints.
* **Cloud Deployment**: Package the API and ML artifacts for hosted deployment.

## 👨‍💻 Author
**[Aryan Singh]**  
*Panipat Institue Of Engineering and Technology*  
*B.Tech Cse(Ai&Ds)*
