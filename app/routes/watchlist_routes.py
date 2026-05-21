from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from .. import models, schemas, auth, database

router = APIRouter(prefix="/watchlist", tags=["watchlist"])

@router.post("/add")
def add_to_watchlist(
    data: schemas.WatchlistAdd,
    current_user: models.User = Depends(auth.get_current_user_from_token),
    db: Session = Depends(database.get_db)
):
    # Check if already in watchlist
    existing = db.query(models.Watchlist).filter(
        models.Watchlist.user_id == current_user.id,
        models.Watchlist.movie_id == data.movie_id
    ).first()
    if existing:
        return {"message": "Movie already in watchlist"}
    
    new_entry = models.Watchlist(
        user_id=current_user.id,
        movie_id=data.movie_id
    )
    db.add(new_entry)
    db.commit()
    return {"message": "Added to watchlist"}

@router.delete("/remove/{movie_id}")
def remove_from_watchlist(
    movie_id: int,
    current_user: models.User = Depends(auth.get_current_user_from_token),
    db: Session = Depends(database.get_db)
):
    entry = db.query(models.Watchlist).filter(
        models.Watchlist.user_id == current_user.id,
        models.Watchlist.movie_id == movie_id
    ).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Movie not found in watchlist")
    
    db.delete(entry)
    db.commit()
    return {"message": "Removed from watchlist"}

@router.get("/")
def get_watchlist(
    current_user: models.User = Depends(auth.get_current_user_from_token),
    db: Session = Depends(database.get_db)
):
    watchlist = db.query(models.Watchlist).filter(models.Watchlist.user_id == current_user.id).all()
    return [{"movie_id": item.movie_id, "added_at": item.added_at} for item in watchlist]
