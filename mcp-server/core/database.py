"""PostgreSQL 数据库管理 - 异步版本

所有函数通过 get_connection() 获取 AsyncSession，使用 SQLAlchemy 2.x 风格查询。
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np

logger = logging.getLogger("research-server.db")


def _get_async_session_factory():
    """延迟导入 get_session_factory，避免循环依赖。"""
    from web.backend.db.base import get_session_factory
    return get_session_factory()


@asynccontextmanager
async def get_connection():
    """获取异步 PostgreSQL session 的上下文管理器。

    用法:
        async with get_connection() as conn:
            paper = await get_paper(conn, paper_id)
    """
    factory = _get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Paper CRUD ────────────────────────────────────────────────────────────

async def resolve_paper_id(conn, identifiers: list[str]) -> str | None:
    """Resolve any canonical external identifier to the shared internal paper ID."""
    from core.paper_id import canonicalize_identifiers
    from web.backend.db.models import Paper, PaperIdentifier
    from sqlalchemy import select

    canonical = canonicalize_identifiers(identifiers)
    if not canonical:
        return None

    result = await conn.execute(
        select(PaperIdentifier.identifier, PaperIdentifier.paper_id)
        .join(Paper, Paper.id == PaperIdentifier.paper_id)
        .where(PaperIdentifier.identifier.in_(canonical))
        .where(Paper.deleted_at.is_(None))
    )
    aliases = {identifier: paper_id for identifier, paper_id in result.all()}
    for identifier in canonical:
        if identifier in aliases:
            return aliases[identifier]

    result = await conn.execute(
        select(Paper.id).where(Paper.id.in_(canonical), Paper.deleted_at.is_(None))
    )
    existing = set(result.scalars().all())
    return next((identifier for identifier in canonical if identifier in existing), None)


async def add_paper_identifiers(conn, paper_id: str, identifiers: list[str]) -> None:
    from core.paper_id import canonicalize_identifiers, identifier_type
    from web.backend.db.models import PaperIdentifier

    canonical = canonicalize_identifiers(identifiers)
    if not canonical:
        return

    dialect = conn.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        insert = None

    if insert is not None:
        for identifier in canonical:
            statement = insert(PaperIdentifier).values(
                paper_id=paper_id,
                identifier_type=identifier_type(identifier),
                identifier=identifier,
            ).on_conflict_do_nothing(index_elements=["identifier"])
            await conn.execute(statement)
    else:
        from sqlalchemy import select
        for identifier in canonical:
            result = await conn.execute(
                select(PaperIdentifier).where(PaperIdentifier.identifier == identifier)
            )
            if result.scalar_one_or_none() is None:
                conn.add(PaperIdentifier(
                    paper_id=paper_id,
                    identifier_type=identifier_type(identifier),
                    identifier=identifier,
                ))
    await conn.flush()


async def upsert_paper(conn, paper: dict) -> str:
    """Insert/update paper metadata and return its resolved shared identity."""
    from core.paper_id import canonicalize_identifiers, canonicalize_paper_id
    from web.backend.db.models import Paper
    from sqlalchemy import select

    incoming_id = canonicalize_paper_id(paper["id"])
    identifiers = canonicalize_identifiers([incoming_id, *(paper.get("identifiers") or [])])
    resolved_id = await resolve_paper_id(conn, identifiers) or incoming_id
    paper["id"] = resolved_id

    authors = paper.get("authors", [])
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except (json.JSONDecodeError, TypeError):
            authors = [authors] if authors else []

    result = await conn.execute(select(Paper).where(Paper.id == paper["id"]))
    existing = result.scalar_one_or_none()

    if existing:
        if existing.deleted_at is not None:
            raise RuntimeError(f"Paper {existing.id} is being cleaned up")
        existing.title = paper["title"]
        existing.authors = authors
        if paper.get("abstract"):
            existing.abstract = paper["abstract"]
        if paper.get("year"):
            existing.year = paper["year"]
        if paper.get("source") and (
            not existing.source
            or existing.source == "parsed"
            or paper["source"] not in {"parsed", "upload"}
        ):
            existing.source = paper["source"]
        if paper.get("pdf_path"):
            existing.pdf_path = paper["pdf_path"]
        if paper.get("url"):
            existing.url = paper["url"]
        if paper.get("citation_count"):
            existing.citation_count = max(
                existing.citation_count or 0, paper["citation_count"]
            )
        if paper.get("visibility") == "public":
            existing.visibility = "public"
        if paper.get("created_by_user_id") and not existing.created_by_user_id:
            existing.created_by_user_id = paper["created_by_user_id"]
    else:
        conn.add(Paper(
            id=paper["id"],
            title=paper["title"],
            authors=authors,
            abstract=paper.get("abstract"),
            year=paper.get("year"),
            source=paper.get("source"),
            pdf_path=paper.get("pdf_path"),
            url=paper.get("url"),
            citation_count=paper.get("citation_count", 0),
            visibility=paper.get("visibility", "public"),
            created_by_user_id=paper.get("created_by_user_id"),
        ))
    await conn.flush()
    await add_paper_identifiers(conn, resolved_id, identifiers)
    return resolved_id


async def get_paper(conn, paper_id: str) -> Optional[dict]:
    """获取单篇论文，返回 dict 或 None"""
    from web.backend.db.models import Paper
    from sqlalchemy import select

    result = await conn.execute(
        select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None))
    )
    row = result.scalar_one_or_none()
    if not row:
        return None
    return {
        "id": row.id,
        "title": row.title,
        "authors": row.authors if isinstance(row.authors, list) else (json.loads(row.authors) if row.authors else []),
        "abstract": row.abstract,
        "year": row.year,
        "source": row.source,
        "pdf_path": row.pdf_path,
        "url": row.url,
        "citation_count": row.citation_count,
        "visibility": row.visibility,
        "created_by_user_id": str(row.created_by_user_id) if row.created_by_user_id else None,
        "created_at": str(row.created_at) if row.created_at else None,
    }


async def get_all_papers(conn) -> list[dict]:
    """获取所有论文"""
    from web.backend.db.models import Paper
    from sqlalchemy import select

    result = await conn.execute(
        select(Paper).where(Paper.deleted_at.is_(None)).order_by(Paper.created_at.desc())
    )
    rows = result.scalars().all()
    return [{
        "id": r.id, "title": r.title, "authors": r.authors,
        "abstract": r.abstract, "year": r.year, "source": r.source,
        "pdf_path": r.pdf_path, "url": r.url, "citation_count": r.citation_count,
    } for r in rows]


# ── Chunk CRUD ────────────────────────────────────────────────────────────

async def insert_chunks(conn, paper_id: str, chunks: list[dict]) -> list[int]:
    """批量插入分块，返回 chunk_id 列表"""
    from web.backend.db.models import Chunk

    ids = []
    for chunk in chunks:
        obj = Chunk(
            paper_id=paper_id,
            section=chunk.get("section"),
            ordinal=chunk.get("ordinal", 0),
            text=chunk["text"],
        )
        conn.add(obj)
        await conn.flush()
        ids.append(obj.id)
    return ids


async def get_chunks_by_paper(conn, paper_id: str) -> list[dict]:
    """获取论文的所有分块"""
    from web.backend.db.models import Chunk
    from sqlalchemy import select

    result = await conn.execute(
        select(Chunk)
        .where(Chunk.paper_id == paper_id)
        .order_by(Chunk.ordinal)
    )
    rows = result.scalars().all()
    return [{"id": r.id, "paper_id": r.paper_id, "section": r.section,
             "ordinal": r.ordinal, "text": r.text} for r in rows]


async def get_all_chunks(conn) -> list[dict]:
    """获取所有分块"""
    from web.backend.db.models import Chunk
    from sqlalchemy import select

    result = await conn.execute(select(Chunk).order_by(Chunk.paper_id, Chunk.ordinal))
    rows = result.scalars().all()
    return [{"id": r.id, "paper_id": r.paper_id, "section": r.section,
             "ordinal": r.ordinal, "text": r.text} for r in rows]


# ── Embedding CRUD ────────────────────────────────────────────────────────

async def insert_embeddings_batch(conn, chunk_ids: list[int], vecs: list[list[float]]) -> None:
    """批量插入向量"""
    from web.backend.db.models import Embedding

    for chunk_id, vec in zip(chunk_ids, vecs):
        arr = np.array(vec, dtype=np.float32)
        conn.add(Embedding(
            chunk_id=chunk_id,
            dim=len(arr),
            vec=arr.tobytes(),
        ))
    await conn.flush()


async def get_all_embeddings(conn) -> list[dict]:
    """获取所有向量（用于 brute-force cosine similarity 检索）"""
    from web.backend.db.models import Embedding, Chunk, Paper
    from sqlalchemy import select

    result = await conn.execute(
        select(Embedding, Chunk, Paper)
        .join(Chunk, Embedding.chunk_id == Chunk.id)
        .join(Paper, Chunk.paper_id == Paper.id)
        .where(Paper.deleted_at.is_(None))
    )
    rows = result.all()
    results = []
    for emb, chunk, paper in rows:
        vec = np.frombuffer(emb.vec, dtype=np.float32).copy()
        results.append({
            "chunk_id": emb.chunk_id,
            "dim": emb.dim,
            "vec": vec,
            "text": chunk.text,
            "section": chunk.section,
            "paper_id": chunk.paper_id,
            "title": paper.title,
        })
    return results


async def get_embeddings_by_paper_ids(conn, paper_ids: set[str] | list[str]) -> list[dict]:
    """仅获取指定 paper_ids 对应的向量"""
    from web.backend.db.models import Embedding, Chunk, Paper
    from sqlalchemy import select

    if not paper_ids:
        return []

    paper_id_list = list(paper_ids)
    result = await conn.execute(
        select(Embedding, Chunk, Paper)
        .join(Chunk, Embedding.chunk_id == Chunk.id)
        .join(Paper, Chunk.paper_id == Paper.id)
        .where(Chunk.paper_id.in_(paper_id_list), Paper.deleted_at.is_(None))
    )
    rows = result.all()
    results = []
    for emb, chunk, paper in rows:
        vec = np.frombuffer(emb.vec, dtype=np.float32).copy()
        results.append({
            "chunk_id": emb.chunk_id,
            "dim": emb.dim,
            "vec": vec,
            "text": chunk.text,
            "section": chunk.section,
            "paper_id": chunk.paper_id,
            "title": paper.title,
        })
    return results


# ── Citation CRUD ─────────────────────────────────────────────────────────

async def insert_citation(conn, source_id: str, target_id: str) -> None:
    """插入引用关系（忽略重复）"""
    from web.backend.db.models import Citation
    from sqlalchemy import select

    result = await conn.execute(
        select(Citation).where(
            Citation.source_id == source_id,
            Citation.target_id == target_id,
        )
    )
    existing = result.scalar_one_or_none()
    if not existing:
        conn.add(Citation(source_id=source_id, target_id=target_id))
        await conn.flush()


async def get_citations(conn, paper_id: str) -> dict:
    """获取论文的引用关系"""
    from web.backend.db.models import Citation
    from sqlalchemy import select

    result_citing = await conn.execute(
        select(Citation.target_id).where(Citation.source_id == paper_id)
    )
    citing = result_citing.all()

    result_cited_by = await conn.execute(
        select(Citation.source_id).where(Citation.target_id == paper_id)
    )
    cited_by = result_cited_by.all()

    return {
        "citing": [r[0] for r in citing],
        "cited_by": [r[0] for r in cited_by],
    }
