"""论文查询端点"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.base import get_db
from web.backend.db.models import User, Chunk
from web.backend.auth.dependencies import get_current_user
from web.backend.auth.service import decode_access_token
from web.backend.repositories import PaperRepository, UserPaperRepository
from web.backend.models import PaperFullTextOut, PaperOut
from web.backend.paper_cleanup import process_cleanup_job, schedule_paper_cleanup

logger = logging.getLogger("novare.web.papers")
router = APIRouter(prefix="/api/papers", tags=["papers"])

_optional_bearer = HTTPBearer(auto_error=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    """判断 path 是否在 root 之下（兼容 Python <3.9）。"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _public_papers_dir() -> Path:
    """全局公共 PDF 缓存目录（与 mcp-server/tools/paper_parse.py 保持一致）。"""
    return Path(os.environ.get("RESEARCH_DATA_DIR", "./data")) / "public_papers"


def _is_public_cached_pdf(pdf_path: str) -> bool:
    """判断 pdf_path 是否位于公共论文缓存目录下。"""
    try:
        return _is_relative_to(Path(pdf_path), _public_papers_dir())
    except Exception:
        return False


async def _can_view_local_pdf(paper, user: User, user_paper_repo: UserPaperRepository) -> bool:
    """判断用户是否有权读取本地 PDF 文件。"""
    if paper.created_by_user_id == user.id:
        return True
    if await user_paper_repo.has_fulltext_access(paper.id):
        return True
    if paper.visibility == "public" and paper.pdf_path and _is_public_cached_pdf(paper.pdf_path):
        return True
    return False


def _paper_to_out(paper, is_parsed: bool) -> dict:
    """将 ORM Paper 对象转为 PaperOut dict"""
    authors = paper.authors if isinstance(paper.authors, list) else []
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": authors,
        "abstract": paper.abstract,
        "year": paper.year,
        "source": paper.source,
        "url": paper.url,
        "pdf_path": paper.pdf_path,
        "citation_count": paper.citation_count or 0,
        "is_parsed": is_parsed,
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


async def _can_view_paper_text(paper, user: User, user_paper_repo: UserPaperRepository) -> bool:
    return (
        paper.created_by_user_id == user.id
        or await user_paper_repo.has_fulltext_access(paper.id)
    )


async def _get_ordered_chunks(db: AsyncSession, paper_id: str) -> list[Chunk]:
    result = await db.execute(
        select(Chunk)
        .where(Chunk.paper_id == paper_id)
        .order_by(Chunk.ordinal)
    )
    return list(result.scalars().all())


def _chunks_to_fulltext(paper, chunks: list[Chunk]) -> dict:
    sections_by_name: dict[str, dict] = {}
    for chunk in chunks:
        section_name = (chunk.section or "").strip() or "正文"
        section = sections_by_name.setdefault(
            section_name,
            {"section": section_name, "parts": [], "chunk_count": 0},
        )
        text = (chunk.text or "").strip()
        if text:
            section["parts"].append(text)
        section["chunk_count"] += 1

    sections = [
        {
            "section": section["section"],
            "text": "\n\n".join(section["parts"]).strip(),
            "chunk_count": section["chunk_count"],
        }
        for section in sections_by_name.values()
    ]
    content = "\n\n".join(
        f"## {section['section']}\n\n{section['text']}".strip()
        for section in sections
    )
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "sections": sections,
        "content": content,
    }


async def _try_get_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_optional_bearer),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """可选认证：有 token 则解析，没有则返回 None。"""
    if credentials is None:
        return None
    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        return None
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active.is_(True))
    )
    return result.scalar_one_or_none()


# ── 无认证端点（放在 {paper_id:path} 之前，避免路径被吞） ──


@router.get("/pdf/view")
async def get_paper_pdf(
    paper_id: str = Query(..., description="论文 ID"),
    user: User | None = Depends(_try_get_user),
    db: AsyncSession = Depends(get_db),
):
    """获取论文 PDF"""
    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if paper.pdf_path and os.path.isfile(paper.pdf_path):
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required to access local PDF")
        user_paper_repo = UserPaperRepository(db, user.id)
        if not await _can_view_local_pdf(paper, user, user_paper_repo):
            raise HTTPException(status_code=403, detail="You do not have access to this paper's PDF")
        return FileResponse(
            paper.pdf_path,
            media_type="application/pdf",
            filename=f"{paper.title[:80]}.pdf",
        )

    pid = paper.id
    if pid.startswith("arxiv:"):
        arxiv_id = pid.removeprefix("arxiv:")
        return RedirectResponse(url=f"https://arxiv.org/pdf/{arxiv_id}", status_code=302)

    if pid.startswith("doi:10.48550/arXiv."):
        arxiv_id = pid.removeprefix("doi:10.48550/arXiv.")
        return RedirectResponse(url=f"https://arxiv.org/pdf/{arxiv_id}", status_code=302)

    if paper.url:
        return RedirectResponse(url=paper.url, status_code=302)

    raise HTTPException(status_code=404, detail="PDF not available")


# ── 需要认证的端点 ──


@router.get("", response_model=list[PaperOut])
async def list_papers(
    q: str | None = Query(None, description="搜索关键词"),
    is_parsed: bool | None = Query(None, description="是否已解析"),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出当前用户的论文"""
    paper_repo = PaperRepository(db)
    user_paper_repo = UserPaperRepository(db, user.id)

    user_paper_ids = set(await user_paper_repo.get_user_papers())
    if not user_paper_ids:
        return []

    fulltext_ids = await user_paper_repo.get_fulltext_paper_ids()
    papers = await paper_repo.get_all(q=q, user_id=user.id)

    result = []
    for p in papers:
        if p.id not in user_paper_ids:
            continue
        parsed = p.id in fulltext_ids
        if is_parsed is not None and parsed != is_parsed:
            continue
        result.append(_paper_to_out(p, parsed))
        if len(result) >= limit:
            break

    return result


@router.get("/{paper_id:path}/chunks")
async def get_paper_chunks(
    paper_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取论文的文本块"""
    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_visible(paper_id, user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    user_paper_repo = UserPaperRepository(db, user.id)
    if not await _can_view_paper_text(paper, user, user_paper_repo):
        raise HTTPException(status_code=403, detail="You do not have access to this paper's chunks")

    chunks = await _get_ordered_chunks(db, paper_id)
    return [
        {"id": c.id, "section": c.section, "ordinal": c.ordinal, "text": c.text}
        for c in chunks
    ]


@router.get("/{paper_id:path}/fulltext", response_model=PaperFullTextOut)
async def get_paper_fulltext(
    paper_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取已解析论文的完整文本"""
    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_visible(paper_id, user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    user_paper_repo = UserPaperRepository(db, user.id)
    if not await _can_view_paper_text(paper, user, user_paper_repo):
        raise HTTPException(status_code=403, detail="You do not have access to this paper's full text")

    chunks = await _get_ordered_chunks(db, paper_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Full text not available")

    return _chunks_to_fulltext(paper, chunks)


@router.get("/{paper_id:path}", response_model=PaperOut)
async def get_paper(
    paper_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取论文详情"""
    paper_repo = PaperRepository(db)
    user_paper_repo = UserPaperRepository(db, user.id)

    paper = await paper_repo.get_visible(paper_id, user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    is_parsed = await user_paper_repo.has_fulltext_access(paper_id)
    return _paper_to_out(paper, is_parsed)


@router.delete("/{paper_id:path}")
async def delete_paper(
    paper_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """从当前用户的论文库中移除论文。"""
    paper_repo = PaperRepository(db)

    paper = await paper_repo.get_visible(paper_id, user.id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    job = await schedule_paper_cleanup(
        db,
        paper_id=paper_id,
        user_id=user.id,
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Paper not associated with user")
    await db.commit()
    completed = await process_cleanup_job(db, job.id)
    return {
        "ok": True,
        "cleanup_job_id": str(job.id),
        "cleanup_scope": job.scope,
        "cleanup_status": completed.status if completed else "pending",
    }
