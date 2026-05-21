from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas, auth, database

router = APIRouter(prefix="/ratings", tags=["ratings"])

@router.post("/add")
def add_rating(
    data: schemas.RatingAdd,
    current_user: models.User = Depends(auth.get_current_user_from_token),
    db: Session = Depends(database.get_db)
):
    if not (0.5 <= data.rating <= 5.0):
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")
        
    # Check if already rated
    existing = db.query(models.UserRating).filter(
        models.UserRating.user_id == current_user.id,
        models.UserRating.movie_id == data.movie_id
    ).first()
    
    if existing:
        existing.rating = data.rating
        existing.rated_at = models.datetime.utcnow()
        db.commit()
        return {"message": "Rating updated"}
    
    new_rating = models.UserRating(
        user_id=current_user.id,
        movie_id=data.movie_id,
        rating=data.rating
    )
    db.add(new_rating)
    db.commit()
    return {"message": "Rating added"}

@router.get("/")
def get_user_ratings(
    current_user: models.User = Depends(auth.get_current_user_from_token),
    db: Session = Depends(database.get_db)
):
    ratings = db.query(models.UserRating).filter(models.UserRating.user_id == current_user.id).all()
    return [{"movie_id": r.movie_id, "rating": r.rating, "rated_at": r.rated_at} for r in ratings]
