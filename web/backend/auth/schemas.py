from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _uuid_to_str(v: UUID | str) -> str:
    return str(v)


class UserResponse(BaseModel):
    id: Annotated[str, BeforeValidator(_uuid_to_str)]
    username: str
    email: str

    class Config:
        from_attributes = True
