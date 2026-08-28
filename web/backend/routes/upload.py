"""文件上传端点"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from novare.file_storage import store_upload_stream
from web.backend.auth.dependencies import get_current_user
from web.backend.db.base import get_db
from web.backend.db.models import User
from web.backend.models import UploadResponse
from web.backend.repositories.upload_repo import UploadRepository

logger = logging.getLogger("novare.web.upload")
router = APIRouter(prefix="/api/upload", tags=["upload"])


def _safe_filename(name: str) -> str:
    """剥离路径，只保留纯文件名，防止路径穿越。"""
    name = Path(name).name  # 去掉目录部分 (../../etc/passwd → passwd)
    name = re.sub(r"[^\w.\-]", "_", name)  # 仅保留安全字符
    return name or "unnamed"


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Store one global blob and create a user-scoped upload authorization."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    safe_name = _safe_filename(file.filename)
    try:
        stored = await store_upload_stream(file)
        if stored.size_bytes == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        repository = UploadRepository(db, user.id)
        blob = await repository.get_or_create_blob(
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            mime_type=file.content_type,
            storage_path=stored.storage_path,
        )
        upload, created = await repository.get_or_create_upload(
            blob=blob,
            original_filename=safe_name,
        )
        await db.commit()
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.exception("Failed to persist upload for user %s", user.id)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    logger.info(
        "File uploaded by user %s: %s (%d bytes, upload=%s, reused=%s)",
        user.id, safe_name, stored.size_bytes, upload.id, not created,
    )

    return UploadResponse(
        upload_id=str(upload.id),
        filename=safe_name,
        already_uploaded=not created,
        message=f"文件已上传: {safe_name}",
    )
