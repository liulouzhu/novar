"""Content-addressed file storage shared by uploads and paper downloads."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


_READ_SIZE = 1024 * 1024


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class StoredBlob:
    sha256: str
    size_bytes: int
    storage_path: str


def get_blob_root() -> Path:
    root = Path(os.environ.get("RESEARCH_DATA_DIR", "./data")) / "file_blobs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def blob_path_for_hash(sha256: str) -> Path:
    return get_blob_root() / sha256[:2] / sha256


def delete_stored_blob(storage_path: str) -> None:
    """Delete a content-addressed blob while preventing path traversal."""
    root = get_blob_root()
    path = Path(storage_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete file outside blob storage: {path}") from exc
    path.unlink(missing_ok=True)


async def store_upload_stream(upload: AsyncReadable) -> StoredBlob:
    """Stream an upload into content-addressed storage without loading it into memory."""
    temp_dir = get_blob_root() / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="upload-", dir=temp_dir)
    hasher = hashlib.sha256()
    size = 0

    try:
        with os.fdopen(fd, "wb") as destination:
            while True:
                chunk = await upload.read(_READ_SIZE)
                if not chunk:
                    break
                destination.write(chunk)
                hasher.update(chunk)
                size += len(chunk)

        digest = hasher.hexdigest()
        if size == 0:
            Path(temp_name).unlink(missing_ok=True)
            return StoredBlob(digest, 0, "")
        target = blob_path_for_hash(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            Path(temp_name).unlink(missing_ok=True)
        else:
            os.replace(temp_name, target)
        return StoredBlob(digest, size, str(target.resolve()))
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def hash_file(path: str | Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        while chunk := source.read(_READ_SIZE):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def store_existing_file(path: str | Path, *, move: bool = False) -> StoredBlob:
    """Register a local file in blob storage, optionally moving the source file."""
    source = Path(path).resolve()
    digest, size = hash_file(source)
    target = blob_path_for_hash(digest)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if move and source != target:
            source.unlink(missing_ok=True)
    elif move:
        os.replace(source, target)
    else:
        fd, temp_name = tempfile.mkstemp(prefix="copy-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as destination, source.open("rb") as original:
                while chunk := original.read(_READ_SIZE):
                    destination.write(chunk)
            os.replace(temp_name, target)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise

    return StoredBlob(digest, size, str(target.resolve()))
