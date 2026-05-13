# Movie Recommendation System 🎬

## 📌 Objective
The primary objective of this project is to develop a highly robust, scalable, and fully functional web application that provides personalized movie recommendations. By engineering a hybrid pipeline integrating raw data cleaning, machine learning (content-based filtering), and a high-performance backend, the system allows users to intuitively search for movies, manage personal watchlists, rate films, and discover new content seamlessly.

## ✨ Features
* **AI Recommendation Engine**: Employs deep-learning content-based calculations via Scikit-Learn utilizing Cosine Similarities across genres, cast, crew, and plot.
* **Stateful Secure Authentication**: Integrated user tracking utilizing `passlib` PyBcrypt password hashing and Starlette session-cookie management.
* **Interactive UI**: A fully custom, pristine dark-mode frontend rendering efficiently without heavy Javascript frameworks.
* **Dynamic Media Watchlists**: Logged-in users can securely curate and remove items from personal SQL-backed movie tracking queues natively.
* **Community Rating Aggregation**: Captures and mathematically aggregates exact community ratings alongside official datasets in real-time.

## 🛠️ Technology Stack
* **Backend Framework**: Python (FastAPI)
* **Frontend Templates**: Jinja2 HTML, Custom CSS
* **Machine Learning**: Pandas, Scikit-Learn (TF-IDF Vectorizers)
* **Database Integration**: SQLite, SQLAlchemy ORM
* **Security & Auth**: PyBcrypt, ItsDangerous

## 📂 Dataset Used
This platform actively integrates a custom-trimmed slice of the **TMDB (The Movie Database)** dataset (`TMDB_all_movies.csv`). The CSV undergoes strict data sanitation using the project's native `clean_data.py` pipeline to drop null fields, patch encoding errors, and concatenate "Tags" needed for the AI engine prior to deployment.

## 🏗️ Project Architecture
The application leverages a classic MVC (Model-View-Controller) structure decoupled natively into backend routines and template views.
* **Controllers (`main.py`)**: Manage routing requests, session parsing, and template injection.
* **Models (`models.py / schemas.py`)**: Control data validation structures traversing in and out of the persistent local `.db` file. 
* **Views (`/templates`)**: Renders final states dynamically to web clients.

## 🧩 Core Modules
1. **`clean_data.py`** – Parses raw large CSV constraints securely generating strict structural integrity.
2. **`app/recommender.py`** – Bootloads finalized `.csv` matrices instantly upon server generation returning cosine logic mathematically without lag.
3. **`app/database.py` / `models.py`** – Constructs and bridges active SQLAlchemy user schemas dynamically.

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
Once the log reads that the startup is complete, seamlessly open your browser to **http://127.0.0.1:8000** to interface with the server immediately!

*(Note: Data structure collisions can be fixed by simply deleting the `sql_app.db` file and re-running the command so SQLAlchemy recreates your schemas cleanly.)*

---

## 🔮 Future Scope
* **Collaborative Filtering**: Advancing the current engine strictly from metadata parsing to analyzing large arrays of user-rating comparisons collectively.
* **Cloud Migration**: Stripping SQLite natively out to host persistent AWS PostgreSQL RDS environments securely.
* **API Isolation**: Bridging templates out entirely for a strictly decoupled Next.js or React frontend relying on JWTs.

## 👨‍💻 Author
**[Aryan Singh]**  
*Panipat Institue Of Engineering and Technology*  
*B.Tech Cse(Ai&Ds)*
