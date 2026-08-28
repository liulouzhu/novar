"""tests/test_paper_pdf_access.py — /api/papers/pdf/view 本地 PDF 访问权限测试。

验证：
- _is_public_cached_pdf 正确判断公共缓存目录
- _can_view_local_pdf 权限逻辑与 chunks/fulltext 一致
- paper.visibility == "public" 不再单独授权读取任意本地 pdf_path
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

# ── _is_public_cached_pdf ────────────────────────────────────────────────


class TestIsPublicCachedPdf:
    """pdf_path 在 public_papers 目录下才返回 True。"""

    def test_path_in_public_dir(self, tmp_path):
        public_dir = tmp_path / "public_papers"
        public_dir.mkdir()
        pdf = public_dir / "arxiv_1234.pdf"
        pdf.write_bytes(b"%PDF")

        from web.backend.routes.papers import _is_public_cached_pdf
        with patch("web.backend.routes.papers._public_papers_dir", return_value=public_dir):
            assert _is_public_cached_pdf(str(pdf)) is True

    def test_path_in_subdir_of_public(self, tmp_path):
        public_dir = tmp_path / "public_papers"
        sub = public_dir / "arxiv_1234"
        sub.mkdir(parents=True)
        pdf = sub / "paper.pdf"
        pdf.write_bytes(b"%PDF")

        from web.backend.routes.papers import _is_public_cached_pdf
        with patch("web.backend.routes.papers._public_papers_dir", return_value=public_dir):
            assert _is_public_cached_pdf(str(pdf)) is True

    def test_path_in_user_workspace(self, tmp_path):
        public_dir = tmp_path / "public_papers"
        public_dir.mkdir()
        user_pdf = tmp_path / "workspace" / "userA" / "uploads" / "paper.pdf"
        user_pdf.parent.mkdir(parents=True)
        user_pdf.write_bytes(b"%PDF")

        from web.backend.routes.papers import _is_public_cached_pdf
        with patch("web.backend.routes.papers._public_papers_dir", return_value=public_dir):
            assert _is_public_cached_pdf(str(user_pdf)) is False

    def test_none_path(self):
        from web.backend.routes.papers import _is_public_cached_pdf
        assert _is_public_cached_pdf(None) is False

    def test_empty_path(self):
        from web.backend.routes.papers import _is_public_cached_pdf
        assert _is_public_cached_pdf("") is False


# ── _can_view_local_pdf ──────────────────────────────────────────────────


class TestCanViewLocalPdf:
    """本地 PDF 访问权限：owner / fulltext_access / public_cached 三种放行路径。"""

    @pytest.mark.asyncio
    async def test_owner_can_view(self):
        """paper.created_by_user_id == user.id → 允许。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "user-1"
        paper.visibility = "private"
        paper.pdf_path = "/some/path.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "user-1"

        repo = AsyncMock()
        assert await _can_view_local_pdf(paper, user, repo) is True

    @pytest.mark.asyncio
    async def test_fulltext_access_can_view(self):
        """UserPaper.has_fulltext_access=True → 允许。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "other-user"
        paper.visibility = "private"
        paper.pdf_path = "/some/path.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "user-1"

        repo = AsyncMock()
        repo.has_fulltext_access = AsyncMock(return_value=True)
        assert await _can_view_local_pdf(paper, user, repo) is True

    @pytest.mark.asyncio
    async def test_public_cached_pdf_can_view(self):
        """public paper + pdf_path 在 public_papers 目录 → 允许。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "other-user"
        paper.visibility = "public"
        paper.pdf_path = "/data/public_papers/arxiv_1234.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "user-1"

        repo = AsyncMock()
        repo.has_fulltext_access = AsyncMock(return_value=False)
        with patch("web.backend.routes.papers._is_public_cached_pdf", return_value=True):
            assert await _can_view_local_pdf(paper, user, repo) is True

    @pytest.mark.asyncio
    async def test_public_but_user_local_pdf_denied(self):
        """public paper + pdf_path 在用户目录 → 无 fulltext access 时拒绝。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "other-user"
        paper.visibility = "public"
        paper.pdf_path = "/workspace/userA/uploads/paper.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "user-1"

        repo = AsyncMock()
        repo.has_fulltext_access = AsyncMock(return_value=False)
        with patch("web.backend.routes.papers._is_public_cached_pdf", return_value=False):
            assert await _can_view_local_pdf(paper, user, repo) is False

    @pytest.mark.asyncio
    async def test_private_non_owner_no_access_denied(self):
        """private paper + 非 owner + 无 fulltext access → 拒绝。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "owner"
        paper.visibility = "private"
        paper.pdf_path = "/some/path.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "intruder"

        repo = AsyncMock()
        repo.has_fulltext_access = AsyncMock(return_value=False)
        assert await _can_view_local_pdf(paper, user, repo) is False

    @pytest.mark.asyncio
    async def test_public_user_pdf_with_fulltext_access_can_view(self):
        """public paper + pdf_path 在用户目录 + 有 fulltext access → 允许。"""
        from web.backend.routes.papers import _can_view_local_pdf
        paper = MagicMock()
        paper.created_by_user_id = "other-user"
        paper.visibility = "public"
        paper.pdf_path = "/workspace/userA/uploads/paper.pdf"
        paper.id = "p1"

        user = MagicMock()
        user.id = "user-1"

        repo = AsyncMock()
        repo.has_fulltext_access = AsyncMock(return_value=True)
        assert await _can_view_local_pdf(paper, user, repo) is True


# ── _is_relative_to ──────────────────────────────────────────────────────


class TestIsRelativeTo:
    def test_child_in_root(self, tmp_path):
        from web.backend.routes.papers import _is_relative_to
        root = tmp_path / "root"
        child = root / "sub" / "file.txt"
        assert _is_relative_to(child, root) is True

    def test_outside_root(self, tmp_path):
        from web.backend.routes.papers import _is_relative_to
        root = tmp_path / "root"
        other = tmp_path / "other" / "file.txt"
        assert _is_relative_to(other, root) is False

    def test_root_itself(self, tmp_path):
        from web.backend.routes.papers import _is_relative_to
        root = tmp_path / "root"
        assert _is_relative_to(root, root) is True
