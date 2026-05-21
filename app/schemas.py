from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str

class WatchlistAdd(BaseModel):
    movie_id: int

class RatingAdd(BaseModel):
    movie_id: int
    rating: float
