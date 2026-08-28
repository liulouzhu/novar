import os
import secrets
import logging
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_secret = os.getenv("JWT_SECRET_KEY")
if not _secret:
    _secret = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY not set — generated ephemeral key. "
        "Set JWT_SECRET_KEY env var for persistent sessions across restarts."
    )
SECRET_KEY = _secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
