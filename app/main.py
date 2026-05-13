from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware
import hashlib
import os
import urllib.parse

from . import models, schemas, database, recommender
from .models import User, Watchlist

from .database import engine
from .models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommendation System")

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
