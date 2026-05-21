from pydantic import BaseModel, EmailStr
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    
    class Config:
        from_attributes = True

# --- Watchlist Schemas ---
class WatchlistBase(BaseModel):
    movie_title: str

class WatchlistCreate(WatchlistBase):
    pass

class Watchlist(WatchlistBase):
    id: int
    user_id: int
    added_at: datetime
    
    class Config:
        from_attributes = True

# --- Rating Schemas ---
class RatingBase(BaseModel):
    movie_title: str
    rating: int

class RatingCreate(RatingBase):
    pass

class Rating(RatingBase):
    id: int
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
