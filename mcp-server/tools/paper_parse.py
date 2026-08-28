"""PDF 解析工具 - 下载、解析、分块、向量化、写入数据库"""

import logging
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

import httpx

from core.database import (
    get_connection,
    get_paper,
    get_chunks_by_paper,
    insert_chunks,
    insert_embeddings_batch,
    insert_citation,
    add_paper_identifiers,
    resolve_paper_id,
    upsert_paper,
)
from core.embedding import embed_batch_async, get_embedding_dim, EmbeddingProviderError
from core.paper_id import (
    canonicalize_identifiers,
    canonicalize_paper_id,
    extract_document_identifiers,
)
from core.pdf_parser import (
    split_into_sections,
    chunk_text,
    extract_references,
    extract_paper_ids_from_refs,
)
from core.mineru import parse_pdf_with_mineru
from novare.file_storage import store_existing_file
from tools.result import ok, fail, truncate, MAX_SECTION_PREVIEW, MAX_SECTIONS, MAX_REFS, MAX_REF_LEN

logger = logging.getLogger("research-server.paper_parse")

_GLOBAL_PAPERS_DIR = os.path.join(
    os.environ.get("RESEARCH_DATA_DIR", "./data"), "public_papers",
)
DEFAULT_USER_ID = os.getenv("RAG_DEFAULT_USER", "default")


def _public_papers_dir() -> str:
    """全局公共 PDF 缓存（arxiv / S2 下载的论文）。"""
    return _GLOBAL_PAPERS_DIR


def _user_papers_dir(user_id: str) -> str:
    """用户私有 PDF 目录（用户上传的论文）。"""
    from novare.config import get_user_workspace
    return os.path.join(get_user_workspace(user_id), "papers")


def _resolve_user_workspace(user_id: str) -> Path:
    """返回用户 workspace 的 resolved Path。"""
    from novare.config import get_user_workspace
    return Path(get_user_workspace(user_id)).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    """判断 path 是否在 root 之下（兼容 Python <3.9）。"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_user_local_file(file_path: str, user_id: str | None) -> str:
    """校验本地文件路径属于当前用户的允许目录。

    返回 resolved 路径字符串。
    Raises:
        PermissionError — 无权限或缺少 user context
        FileNotFoundError — 文件不存在
    """
    resolved = Path(file_path).resolve()

    if not user_id:
        if os.getenv("ALLOW_UNSCOPED_LOCAL_FILE_PARSE", "").lower() not in ("1", "true", "yes"):
            raise PermissionError("缺少用户上下文，无法解析本地文件。请在 Web 模式下使用。")
        if not resolved.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return str(resolved)

    user_root = _resolve_user_workspace(user_id)
    allowed_roots = [
        (user_root / "uploads").resolve(),
        (user_root / "papers").resolve(),
    ]

    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise PermissionError("文件路径不在您的允许目录内，无法访问。")

    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return str(resolved)


async def _user_has_fulltext_access(user_id: str, paper_id: str) -> bool:
    """查询用户是否对指定论文有全文访问权限。"""
    try:
        from web.backend.db.base import get_session_factory
        from web.backend.db.models import UserPaper
        from uuid import UUID
        from sqlalchemy import select
        async with get_session_factory()() as db:
            result = await db.execute(
                select(UserPaper).where(
                    UserPaper.user_id == UUID(user_id),
                    UserPaper.paper_id == paper_id,
                    UserPaper.has_fulltext_access.is_(True),
                    UserPaper.deleted_at.is_(None),
                )
            )
            return result.scalar_one_or_none() is not None
    except Exception:
        return False


async def _can_reuse_paper_pdf(paper: dict, user_id: str | None) -> bool:
    """判断当前用户是否可以复用已有 paper 的 pdf_path。

    - public paper + pdf_path 在公共缓存目录 → 允许
    - public paper + pdf_path 在用户目录 → 需要权限
    - private paper → 仅 owner 或 has_fulltext_access
    - 无 pdf_path → False
    """
    pdf_path = paper.get("pdf_path")
    if not pdf_path:
        return False

    visibility = paper.get("visibility", "public")
    creator = str(paper.get("created_by_user_id") or "")
    paper_id = paper["id"]

    if visibility == "private":
        if not user_id:
            return False
        if creator and creator == str(user_id):
            return True
        return await _user_has_fulltext_access(user_id, paper_id)

    # public paper
    public_dir = Path(_public_papers_dir()).resolve()
    pdf_resolved = Path(pdf_path).resolve()
    if _is_relative_to(pdf_resolved, public_dir):
        return True

    # pdf_path 在某个用户目录下 — 需要权限
    if not user_id:
        return False
    if creator and creator == str(user_id):
        return True
    return await _user_has_fulltext_access(user_id, paper_id)


async def associate_user_paper(
    user_id: str,
    paper_id: str,
    relation_type: str = "parsed",
    has_fulltext_access: bool = True,
    source: str = "paper_parse",
):
    """Create or upgrade a user-paper association in PostgreSQL."""
    if not user_id:
        return
    try:
        from web.backend.db.base import get_session_factory
        from web.backend.repositories.user_paper_repo import UserPaperRepository
        from uuid import UUID
        async with get_session_factory()() as db:
            repo = UserPaperRepository(db, UUID(user_id))
            await repo.associate(
                paper_id,
                relation_type=relation_type,
                has_fulltext_access=has_fulltext_access,
                source=source,
            )
            await db.commit()
    except Exception as e:
        logger.warning("Failed to associate user-paper: %s", e)


async def _download_pdf(url: str, dest_path: str) -> bool:
    """下载 PDF 文件"""
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
        logger.info("Downloaded PDF: %s -> %s", url, dest_path)
        return True
    except Exception as e:
        logger.error("Failed to download PDF from %s: %s", url, e)
        return False


async def _get_owned_upload(upload_id: str, user_id: str | None):
    if not user_id:
        raise PermissionError("缺少用户上下文，无法访问上传文件。")
    try:
        parsed_upload_id = UUID(upload_id)
        parsed_user_id = UUID(user_id)
    except (TypeError, ValueError):
        raise PermissionError("无效的 upload_id。")

    from web.backend.db.base import get_session_factory
    from web.backend.repositories.upload_repo import UploadRepository

    async with get_session_factory()() as db:
        owned = await UploadRepository(db, parsed_user_id).get_owned(parsed_upload_id)
    if owned is None:
        raise PermissionError("上传文件不存在或不属于当前用户。")
    if not Path(owned.storage_path).is_file():
        raise FileNotFoundError("上传文件内容不存在。")
    return owned


async def _register_blob(path: str, mime_type: str | None, *, move: bool = False):
    stored = store_existing_file(path, move=move)
    from web.backend.repositories.upload_repo import get_or_create_file_blob
    async with get_connection() as conn:
        blob = await get_or_create_file_blob(
            conn,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            mime_type=mime_type,
            storage_path=stored.storage_path,
        )
        blob_id = blob.id
    return blob_id, stored


def _milvus_insert(
    paper_id: str,
    chunk_ids: list[int],
    chunks: list[dict],
    embeddings: list[list[float]],
    user_id: str = None,
) -> None:
    """Insert embeddings into Milvus. Silently skips on failure."""
    try:
        from core.vector_store import insert_vectors

        texts = [c["text"] for c in chunks]
        insert_vectors(user_id or DEFAULT_USER_ID, paper_id, chunk_ids, texts, embeddings)
    except Exception as e:
        logger.warning("Failed to insert vectors into Milvus (non-fatal): %s", e)


async def _ensure_user_milvus_vectors(user_id: str | None, paper_id: str) -> None:
    """Reuse PostgreSQL embeddings when a new user gains access to parsed chunks."""
    if not user_id or await _user_has_fulltext_access(user_id, paper_id):
        return
    from core.database import get_embeddings_by_paper_ids
    async with get_connection() as conn:
        rows = await get_embeddings_by_paper_ids(conn, {paper_id})
    if not rows:
        return
    chunks = [{"text": row["text"], "section": row["section"]} for row in rows]
    embeddings = [row["vec"].tolist() for row in rows]
    _milvus_insert(
        paper_id,
        [row["chunk_id"] for row in rows],
        chunks,
        embeddings,
        user_id=user_id,
    )


async def handle_paper_parse(args: dict, user_id: str = None) -> str:
    """解析论文 PDF"""
    paper_id = args.get("paper_id")
    pdf_url = args.get("pdf_url")
    file_path = args.get("file_path")
    upload_id = args.get("upload_id")

    # 规范化 paper_id（arXiv ID 统一格式）
    if paper_id:
        paper_id = canonicalize_paper_id(paper_id)

    if not paper_id and not pdf_url and not file_path and not upload_id:
        return fail("paper_parse", "请提供 paper_id、pdf_url、upload_id 或 file_path。")

    # 公共下载目录（所有用户共享）
    public_dir = _public_papers_dir()
    os.makedirs(public_dir, exist_ok=True)

    # 确定 PDF 来源
    pdf_path = None
    resolved_paper_id = paper_id
    is_local_file = False
    owns_upload = False
    blob_id = None
    blob_sha256 = None
    paper_file_source = "download"
    paper_file_access_scope = "public"
    identity_candidates = [paper_id] if paper_id else []
    blob_linked = False

    if paper_id:
        async with get_connection() as conn:
            resolved_paper_id = await resolve_paper_id(conn, [paper_id]) or paper_id

    # Web uploads are authorized by opaque upload_id, never by a client-provided path.
    if upload_id:
        try:
            owned_upload = await _get_owned_upload(upload_id, user_id)
        except (PermissionError, FileNotFoundError) as e:
            return fail("paper_parse", str(e))
        pdf_path = owned_upload.storage_path
        blob_id = owned_upload.blob_id
        blob_sha256 = owned_upload.sha256
        is_local_file = True
        owns_upload = True
        paper_file_source = "upload"
        paper_file_access_scope = "private"

        from web.backend.repositories.upload_repo import get_paper_files_for_blob
        async with get_connection() as conn:
            linked_files = await get_paper_files_for_blob(conn, blob_id)
        if linked_files:
            explicit_match = next(
                (item for item in linked_files if item.paper_id == resolved_paper_id),
                None,
            )
            resolved_paper_id = (explicit_match or linked_files[0]).paper_id
            blob_linked = True

    # 本地文件（用户上传）→ 路径安全校验
    if file_path and not upload_id:
        try:
            pdf_path = _validate_user_local_file(file_path, user_id)
        except PermissionError as e:
            return fail("paper_parse", str(e))
        except FileNotFoundError as e:
            return fail("paper_parse", str(e))
        is_local_file = True
        paper_file_source = "legacy_upload"
        paper_file_access_scope = "private"
        try:
            blob_id, stored = await _register_blob(pdf_path, "application/pdf")
            blob_sha256 = stored.sha256
            pdf_path = stored.storage_path
        except Exception as e:
            return fail("paper_parse", f"无法登记上传文件: {e}")

    async with get_connection() as conn:
        # 尝试从数据库获取论文信息
        if resolved_paper_id:
            paper = await get_paper(conn, resolved_paper_id)
            if paper:
                if not pdf_path and paper.get("pdf_path") and os.path.exists(paper["pdf_path"]):
                    if not owns_upload and not await _can_reuse_paper_pdf(paper, user_id):
                        return fail("paper_parse", "您无权访问该论文的本地 PDF。")
                    pdf_path = str(Path(paper["pdf_path"]).resolve())
                elif paper.get("source") == "arxiv":
                    arxiv_id = resolved_paper_id.replace("arxiv:", "")
                    pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
                elif paper.get("source") == "semantic_scholar":
                    if not pdf_url:
                        pdf_url = _try_get_s2_pdf_url(resolved_paper_id)

    # 下载 PDF（URL 来源 → 全局公共缓存）
    if not pdf_path:
        if pdf_url:
            safe_id = (resolved_paper_id or "unknown").replace("/", "_").replace(":", "_")
            pdf_path = os.path.join(public_dir, f"{safe_id}.pdf")
            if not os.path.exists(pdf_path):
                success = await _download_pdf(pdf_url, pdf_path)
                if not success:
                    return fail("paper_parse", f"无法从 {pdf_url} 下载 PDF。")
            try:
                blob_id, stored = await _register_blob(pdf_path, "application/pdf", move=True)
                blob_sha256 = stored.sha256
                pdf_path = stored.storage_path
            except Exception as e:
                return fail("paper_parse", f"无法登记下载文件: {e}")
        else:
            return fail("paper_parse", "无法确定 PDF 来源。请提供 pdf_url。")

    if not os.path.exists(pdf_path):
        return fail("paper_parse", f"PDF 文件不存在: {pdf_path}")

    # 检查是否已经解析过
    async with get_connection() as conn:
        if resolved_paper_id and (not owns_upload or blob_linked):
            existing_chunks = await get_chunks_by_paper(conn, resolved_paper_id)
            if existing_chunks:
                # 可见性校验：private 论文需要权限才能关联
                paper = await get_paper(conn, resolved_paper_id)
                if paper and paper.get("visibility") == "private" and not owns_upload:
                    creator = str(paper.get("created_by_user_id") or "")
                    if creator and creator != str(user_id or ""):
                        if not await _user_has_fulltext_access(user_id or "", resolved_paper_id):
                            return fail("paper_parse", f"论文 {resolved_paper_id} 是私有论文，您无权访问。")
                if blob_id is not None:
                    from web.backend.repositories.upload_repo import link_paper_file
                    await link_paper_file(
                        conn,
                        paper_id=resolved_paper_id,
                        blob_id=blob_id,
                        source=paper_file_source,
                        access_scope=paper_file_access_scope,
                    )
                await _ensure_user_milvus_vectors(user_id, resolved_paper_id)
                await associate_user_paper(
                    user_id,
                    resolved_paper_id,
                    relation_type="uploaded" if owns_upload else "parsed",
                    source="upload" if owns_upload else "paper_parse",
                )
                return ok(
                    "paper_parse",
                    {
                        "paper_id": resolved_paper_id,
                        "upload_id": upload_id,
                        "already_parsed": True,
                        "deduplicated": bool(blob_id),
                        "chunk_count": len(existing_chunks),
                    },
                    summary=f"论文 {resolved_paper_id} 已解析（{len(existing_chunks)} 个分块）",
                    warnings=["如需重新解析，请先删除相关数据"],
                )

    # 解析 PDF
    try:
        if is_local_file:
            # 本地文件用 pymupdf4llm
            from core.pdf_parser import parse_pdf_to_markdown
            markdown_text = parse_pdf_to_markdown(pdf_path)
        else:
            # URL 用 MinerU，保存到公共 papers 目录
            save_dir = os.path.join(public_dir, resolved_paper_id or "unknown")
            result = await parse_pdf_with_mineru(pdf_url, save_dir=save_dir)
            if not result.success:
                return fail("paper_parse", f"MinerU 解析失败 - {result.error}")
            markdown_text = result.markdown
            if result.saved_dir:
                logger.info("MinerU output saved to: %s", result.saved_dir)
    except Exception as e:
        return fail("paper_parse", f"PDF 解析失败 - {str(e)}")

    if not markdown_text or len(markdown_text) < 100:
        return fail("paper_parse", "PDF 解析结果为空或过短。")

    extracted_identifiers = extract_document_identifiers(markdown_text)
    identity_candidates = canonicalize_identifiers([
        *extracted_identifiers,
        *([paper_id] if paper_id else []),
    ])

    async with get_connection() as conn:
        identified_paper_id = await resolve_paper_id(conn, identity_candidates)
        if blob_linked:
            if identified_paper_id and identified_paper_id != resolved_paper_id:
                return fail(
                    "paper_parse",
                    "文件指纹关联的论文与提取到的 DOI/arXiv 标识冲突，请人工核验。",
                )
        else:
            resolved_paper_id = (
                identified_paper_id
                or (identity_candidates[0] if identity_candidates else None)
                or (f"upload:sha256:{blob_sha256}" if blob_sha256 else None)
            )

        if not resolved_paper_id:
            return fail("paper_parse", "无法为该 PDF 生成稳定论文标识。")

        existing_chunks = await get_chunks_by_paper(conn, resolved_paper_id)
        if existing_chunks:
            await add_paper_identifiers(conn, resolved_paper_id, identity_candidates)
            if blob_id is not None:
                from web.backend.repositories.upload_repo import link_paper_file
                await link_paper_file(
                    conn,
                    paper_id=resolved_paper_id,
                    blob_id=blob_id,
                    source=paper_file_source,
                    access_scope=paper_file_access_scope,
                )
            await _ensure_user_milvus_vectors(user_id, resolved_paper_id)
            await associate_user_paper(
                user_id,
                resolved_paper_id,
                relation_type="uploaded" if owns_upload else "parsed",
                source="upload" if owns_upload else "paper_parse",
            )
            return ok(
                "paper_parse",
                {
                    "paper_id": resolved_paper_id,
                    "upload_id": upload_id,
                    "already_parsed": True,
                    "deduplicated": True,
                    "chunk_count": len(existing_chunks),
                },
                summary=f"论文 {resolved_paper_id} 已解析，已复用 {len(existing_chunks)} 个分块",
            )

    # 按章节分割
    sections = split_into_sections(markdown_text)

    # 分块
    all_chunks = []
    for sec in sections:
        sec_chunks = chunk_text(sec["text"])
        for i, chunk_text_str in enumerate(sec_chunks):
            all_chunks.append({
                "section": sec["section"],
                "ordinal": i,
                "text": chunk_text_str,
            })

    if not all_chunks:
        return fail("paper_parse", "分块结果为空。")

    # 向量化
    embeddings = None
    try:
        texts = [c["text"] for c in all_chunks]
        embeddings = await embed_batch_async(texts)

        # ── 维度一致性校验 ──
        expected_dim = get_embedding_dim()
        if not embeddings:
            logger.warning("Embedding returned empty list")
            embeddings = None
        elif len(embeddings) != len(all_chunks):
            logger.error(
                "Embedding count mismatch: got %d embeddings for %d chunks. "
                "Refusing to write inconsistent data.", len(embeddings), len(all_chunks)
            )
            embeddings = None
        else:
            # 检查所有向量维度一致
            dims = {len(e) for e in embeddings}
            if len(dims) != 1:
                logger.error(
                    "Inconsistent embedding dimensions within batch: %s. "
                    "Refusing to write inconsistent data.", dims
                )
                embeddings = None
            elif list(dims)[0] != expected_dim:
                actual_dim = list(dims)[0]
                logger.error(
                    "Embedding dimension mismatch: got %d, expected %d "
                    "(provider declares dim=%d). Refusing to write. "
                    "Check DASHSCOPE_API_KEY and EMBEDDING_MODEL.",
                    actual_dim, expected_dim, expected_dim,
                )
                embeddings = None
    except EmbeddingProviderError as e:
        logger.error("No embedding provider available: %s", e)
        return fail("paper_parse", str(e))
    except Exception as e:
        logger.warning("Embedding failed, saving chunks without vectors: %s", e)
        embeddings = None

    # 写入数据库
    async with get_connection() as conn:
        # 确保论文存在
        if resolved_paper_id:
            existing = await get_paper(conn, resolved_paper_id)
            if not existing:
                # 从 PDF 内容提取基本信息
                title = _extract_title_from_markdown(markdown_text)
                paper_data = {
                    "id": resolved_paper_id,
                    "identifiers": identity_candidates,
                    "title": title or resolved_paper_id,
                    "authors": [],
                    "abstract": "",
                    "year": None,
                    "source": "upload" if is_local_file else "parsed",
                    "pdf_path": pdf_path,
                    "url": pdf_url,
                    "citation_count": 0,
                }
                if is_local_file and user_id:
                    paper_data["visibility"] = "private"
                    paper_data["created_by_user_id"] = user_id
                resolved_paper_id = await upsert_paper(conn, paper_data)
            else:
                await add_paper_identifiers(conn, resolved_paper_id, identity_candidates)
                from web.backend.db.models import Paper
                from sqlalchemy import select
                result = await conn.execute(
                    select(Paper).where(
                        Paper.id == resolved_paper_id,
                        Paper.deleted_at.is_(None),
                    )
                )
                existing_row = result.scalar_one_or_none()
                if existing_row:
                    if existing_row.visibility == "private" and existing_row.created_by_user_id and not owns_upload:
                        if str(existing_row.created_by_user_id) != str(user_id or ""):
                            if not await _user_has_fulltext_access(user_id or "", resolved_paper_id):
                                return fail("paper_parse", "您无权更新该私有论文的 PDF 路径。")
                    existing_row.pdf_path = pdf_path

        if blob_id is not None:
            from web.backend.repositories.upload_repo import link_paper_file
            await link_paper_file(
                conn,
                paper_id=resolved_paper_id,
                blob_id=blob_id,
                source=paper_file_source,
                access_scope=paper_file_access_scope,
            )

        chunk_ids = await insert_chunks(conn, resolved_paper_id, all_chunks)

        if embeddings:
            await insert_embeddings_batch(conn, chunk_ids, embeddings)

        refs = extract_references(markdown_text)
        ref_ids = extract_paper_ids_from_refs(refs)
        citations_added = 0
        for ref in ref_ids:
            if ref.get("doi"):
                target_id = f"doi:{ref['doi']}"
            elif ref.get("arxiv_id"):
                target_id = f"arxiv:{ref['arxiv_id']}"
            else:
                continue
            await insert_citation(conn, resolved_paper_id, target_id)
            citations_added += 1

    # 同步写入 Milvus（如果可用）
    if embeddings and chunk_ids:
        try:
            _milvus_insert(resolved_paper_id, chunk_ids, all_chunks, embeddings, user_id=user_id)
        except Exception as e:
            # DimensionMismatchError 等已由 _milvus_insert 内部记录
            logger.warning("Milvus insertion failed (non-fatal): %s", e)

    # 同步写入 Elasticsearch（如果可用）
    es_indexed_count = 0
    es_warnings: list[str] = []
    if chunk_ids and resolved_paper_id:
        try:
            from core.elasticsearch_store import bulk_upsert_chunks
            # 获取论文标题
            _es_title = resolved_paper_id
            async with get_connection() as _es_conn:
                _es_paper = await get_paper(_es_conn, resolved_paper_id)
                if _es_paper:
                    _es_title = _es_paper.get("title", resolved_paper_id)
            es_docs = []
            for i, chunk_id in enumerate(chunk_ids):
                sec = all_chunks[i]["section"] if i < len(all_chunks) else ""
                txt = all_chunks[i]["text"] if i < len(all_chunks) else ""
                es_docs.append({
                    "chunk_id": chunk_id,
                    "paper_id": resolved_paper_id,
                    "title": _es_title,
                    "section": sec,
                    "text": txt,
                })
            es_result = await bulk_upsert_chunks(es_docs)
            es_indexed_count = es_result.get("success", 0)
            if es_result.get("errors"):
                es_warnings.extend([f"ES: {e}" for e in es_result["errors"]])
            if es_indexed_count > 0:
                logger.info("ES indexed %d chunks for paper %s", es_indexed_count, resolved_paper_id)
        except Exception as e:
            es_warnings.append(f"Elasticsearch insertion failed: {e}")
            logger.warning("Elasticsearch insertion failed (non-fatal): %s", e)

    # 关联用户与论文（PostgreSQL）
    if resolved_paper_id:
        await associate_user_paper(
            user_id,
            resolved_paper_id,
            relation_type="uploaded" if owns_upload else "parsed",
            source="upload" if owns_upload else "paper_parse",
        )

    # 自动构建知识图谱（从摘要提取实体）
    kg_result = ""
    if resolved_paper_id:
        try:
            from tools.knowledge_graph import extract_from_abstract_sync
            kg_result = await extract_from_abstract_sync(resolved_paper_id, user_id=user_id)
            logger.info("Knowledge graph updated for %s", resolved_paper_id)
        except Exception as e:
            logger.warning("Knowledge graph extraction failed: %s", e)

    # 提取标题（供 sources 使用）
    title = None
    async with get_connection() as conn:
        if resolved_paper_id:
            paper_row = await get_paper(conn, resolved_paper_id)
            if paper_row:
                title = paper_row.get("title")

    # 构建返回结果
    sections_preview = [
        {"name": sec["section"], "preview": truncate(sec["text"], MAX_SECTION_PREVIEW)}
        for sec in sections[:MAX_SECTIONS]
    ]
    references_preview = [truncate(ref, MAX_REF_LEN) for ref in refs[:MAX_REFS]]

    # 知识图谱结果处理：尝试解析为 dict，失败则保留原始文本
    kg_stats = {}
    kg_summary_text = ""
    if kg_result:
        try:
            import json
            kg_stats = json.loads(kg_result)
        except (json.JSONDecodeError, TypeError):
            kg_summary_text = kg_result

    data = {
        "paper_id": resolved_paper_id,
        "upload_id": upload_id,
        "deduplicated": False,
        "section_count": len(sections),
        "chunk_count": len(all_chunks),
        "embedding_done": bool(embeddings),
        "citations_count": citations_added,
        "elasticsearch_indexed": es_indexed_count,
        "sections_preview": sections_preview,
        "references_preview": references_preview,
        "kg": kg_stats,
    }
    if kg_summary_text:
        data["kg_summary"] = kg_summary_text

    return ok(
        "paper_parse",
        data,
        summary=f"论文 {resolved_paper_id} 解析完成：{len(sections)} 章节, {len(all_chunks)} 分块, {citations_added} 引用",
        sources=[{"id": resolved_paper_id, "title": title or resolved_paper_id}],
        warnings=es_warnings,
    )


def _try_get_s2_pdf_url(paper_id: str) -> Optional[str]:
    """尝试从 Semantic Scholar 获取 PDF URL"""
    s2_id = paper_id.replace("doi:", "").replace("s2:", "")
    try:
        import httpx
        resp = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}",
            params={"fields": "openAccessPdf"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            oa = data.get("openAccessPdf")
            if oa:
                return oa.get("url")
    except Exception:
        pass
    return None


def _extract_title_from_markdown(md: str) -> str:
    """从 Markdown 开头提取标题"""
    for line in md.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and len(line) > 10:
            return line[:200]
    return ""
