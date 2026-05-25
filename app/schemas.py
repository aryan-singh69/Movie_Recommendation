from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)

class UserLogin(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str = Field(min_length=1, max_length=72)

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
