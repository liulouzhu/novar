"""tests/test_paper_parse_access.py — paper_parse 多用户文件路径越权测试。

验证：
- 本地 file_path 只能访问当前用户 workspace 下的文件
- ../ 和 symlink 逃逸被拒绝
- 无 user_id 默认禁止解析本地文件
- paper.pdf_path 复用受权限控制
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# mcp-server/ 不在 sys.path 中，需要手动加入
MCP_ROOT = Path(__file__).resolve().parent.parent / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")


# ── _validate_user_local_file ────────────────────────────────────────────

class TestValidateUserLocalFile:
    """本地文件路径必须限制在用户 uploads/ 或 papers/ 目录下。"""

    def test_user_can_access_own_uploads(self, tmp_path):
        """用户 A 可以解析自己 uploads/ 下的文件。"""
        user_ws = tmp_path / "ws" / "userA"
        uploads = user_ws / "uploads"
        uploads.mkdir(parents=True)
        test_pdf = uploads / "paper.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 fake")

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            result = _validate_user_local_file(str(test_pdf), "userA")
            assert result == str(test_pdf.resolve())

    def test_user_can_access_own_papers(self, tmp_path):
        """用户 A 可以解析自己 papers/ 下的文件。"""
        user_ws = tmp_path / "ws" / "userA"
        papers = user_ws / "papers"
        papers.mkdir(parents=True)
        test_pdf = papers / "old.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 fake")

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            result = _validate_user_local_file(str(test_pdf), "userA")
            assert result == str(test_pdf.resolve())

    def test_user_cannot_access_other_user_uploads(self, tmp_path):
        """用户 A 不能解析用户 B 的 uploads/ 下的文件。"""
        user_a_ws = tmp_path / "ws" / "userA"
        user_b_ws = tmp_path / "ws" / "userB"
        uploads_b = user_b_ws / "uploads"
        uploads_b.mkdir(parents=True)
        test_pdf = uploads_b / "secret.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 secret")

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_a_ws):
            with pytest.raises(PermissionError, match="不在您的允许目录内"):
                _validate_user_local_file(str(test_pdf), "userA")

    def test_dot_dot_escape_rejected(self, tmp_path):
        """../ 路径逃逸被拒绝。"""
        user_ws = tmp_path / "ws" / "userA"
        uploads = user_ws / "uploads"
        uploads.mkdir(parents=True)
        # 尝试用 ../ 访问 userA 的父目录
        escape_path = str(uploads / ".." / ".." / "other.txt")

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            with pytest.raises((PermissionError, FileNotFoundError)):
                _validate_user_local_file(escape_path, "userA")

    def test_symlink_escape_rejected(self, tmp_path):
        """symlink 逃逸被拒绝。"""
        user_ws = tmp_path / "ws" / "userA"
        uploads = user_ws / "uploads"
        uploads.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        outside_file = outside / "escape.pdf"
        outside_file.write_bytes(b"%PDF escape")
        symlink = uploads / "link.pdf"
        try:
            symlink.symlink_to(outside_file)
        except OSError:
            pytest.skip("symlink not supported on this platform")

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            with pytest.raises(PermissionError, match="不在您的允许目录内"):
                _validate_user_local_file(str(symlink), "userA")

    def test_no_user_id_fails_by_default(self, tmp_path):
        """无 user_id 时默认拒绝解析本地文件。"""
        test_pdf = tmp_path / "any.pdf"
        test_pdf.write_bytes(b"%PDF-1.4")

        from tools.paper_parse import _validate_user_local_file
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ALLOW_UNSCOPED_LOCAL_FILE_PARSE", None)
            with pytest.raises(PermissionError, match="缺少用户上下文"):
                _validate_user_local_file(str(test_pdf), None)

    def test_no_user_id_allowed_with_env(self, tmp_path):
        """ALLOW_UNSCOPED_LOCAL_FILE_PARSE=true 时允许无 user_id 解析本地文件。"""
        test_pdf = tmp_path / "cli.pdf"
        test_pdf.write_bytes(b"%PDF-1.4")

        from tools.paper_parse import _validate_user_local_file
        with patch.dict(os.environ, {"ALLOW_UNSCOPED_LOCAL_FILE_PARSE": "true"}):
            result = _validate_user_local_file(str(test_pdf), None)
            assert result == str(test_pdf.resolve())

    def test_file_not_found_raises(self, tmp_path):
        """文件不存在时抛 FileNotFoundError。"""
        user_ws = tmp_path / "ws" / "userA"
        uploads = user_ws / "uploads"
        uploads.mkdir(parents=True)

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            with pytest.raises(FileNotFoundError):
                _validate_user_local_file(str(uploads / "nonexistent.pdf"), "userA")

    def test_directory_not_allowed(self, tmp_path):
        """目录路径被拒绝（必须是文件）。"""
        user_ws = tmp_path / "ws" / "userA"
        uploads = user_ws / "uploads"
        uploads.mkdir(parents=True)

        from tools.paper_parse import _validate_user_local_file
        with patch("tools.paper_parse._resolve_user_workspace", return_value=user_ws):
            with pytest.raises(FileNotFoundError, match="文件不存在"):
                _validate_user_local_file(str(uploads), "userA")


# ── _can_reuse_paper_pdf ─────────────────────────────────────────────────

class TestCanReusePaperPdf:
    """paper.pdf_path 复用必须受权限控制。"""

    @pytest.mark.asyncio
    async def test_public_paper_in_public_dir_allowed(self, tmp_path):
        """public paper 的 pdf_path 在公共缓存目录下 → 允许。"""
        public_dir = tmp_path / "public_papers"
        public_dir.mkdir()
        pdf = public_dir / "arxiv_1234.pdf"
        pdf.write_bytes(b"%PDF")

        from tools.paper_parse import _can_reuse_paper_pdf
        with patch("tools.paper_parse._public_papers_dir", return_value=str(public_dir)):
            paper = {
                "id": "arxiv:1234", "visibility": "public",
                "pdf_path": str(pdf), "created_by_user_id": None,
            }
            assert await _can_reuse_paper_pdf(paper, "any-user") is True

    @pytest.mark.asyncio
    async def test_public_paper_in_user_dir_requires_access(self, tmp_path):
        """public paper 的 pdf_path 在某用户目录下 → 其他用户需要权限。"""
        user_b_ws = tmp_path / "ws" / "userB"
        papers_b = user_b_ws / "papers"
        papers_b.mkdir(parents=True)
        pdf = papers_b / "paper.pdf"
        pdf.write_bytes(b"%PDF")

        from tools.paper_parse import _can_reuse_paper_pdf
        with patch("tools.paper_parse._public_papers_dir", return_value=str(tmp_path / "public_papers")):
            paper = {
                "id": "p1", "visibility": "public",
                "pdf_path": str(pdf), "created_by_user_id": "userB",
            }
            # userA 不是 owner 且无 fulltext access → 拒绝
            with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=False):
                assert await _can_reuse_paper_pdf(paper, "userA") is False
            # userA 有 fulltext access → 允许
            with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=True):
                assert await _can_reuse_paper_pdf(paper, "userA") is True
            # owner → 允许
            assert await _can_reuse_paper_pdf(paper, "userB") is True

    @pytest.mark.asyncio
    async def test_private_paper_only_owner_or_access(self, tmp_path):
        """private paper → 仅 owner 或 has_fulltext_access 用户可复用。"""
        from tools.paper_parse import _can_reuse_paper_pdf
        paper = {
            "id": "priv1", "visibility": "private",
            "pdf_path": "/some/path.pdf", "created_by_user_id": "owner",
        }
        # owner → 允许
        assert await _can_reuse_paper_pdf(paper, "owner") is True
        # 其他用户无权限 → 拒绝
        with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=False):
            assert await _can_reuse_paper_pdf(paper, "intruder") is False
        # 其他用户有 fulltext access → 允许
        with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=True):
            assert await _can_reuse_paper_pdf(paper, "collaborator") is True
        # 无 user_id → 拒绝
        assert await _can_reuse_paper_pdf(paper, None) is False

    @pytest.mark.asyncio
    async def test_no_pdf_path_returns_false(self):
        """无 pdf_path → False。"""
        from tools.paper_parse import _can_reuse_paper_pdf
        paper = {"id": "p1", "visibility": "public", "pdf_path": None}
        assert await _can_reuse_paper_pdf(paper, "user") is False
        paper2 = {"id": "p1", "visibility": "public"}
        assert await _can_reuse_paper_pdf(paper2, "user") is False

    @pytest.mark.asyncio
    async def test_private_paper_no_creator_with_access(self):
        """private paper 无 created_by_user_id → 有 fulltext access 才允许。"""
        from tools.paper_parse import _can_reuse_paper_pdf
        paper = {
            "id": "p1", "visibility": "private",
            "pdf_path": "/some/path.pdf", "created_by_user_id": None,
        }
        with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=True):
            assert await _can_reuse_paper_pdf(paper, "user") is True
        with patch("tools.paper_parse._user_has_fulltext_access", new_callable=AsyncMock, return_value=False):
            assert await _can_reuse_paper_pdf(paper, "user") is False


# ── _is_relative_to ──────────────────────────────────────────────────────

class TestIsRelativeTo:
    def test_child_path(self, tmp_path):
        from tools.paper_parse import _is_relative_to
        root = tmp_path / "root"
        child = root / "sub" / "file.txt"
        assert _is_relative_to(child, root) is True

    def test_outside_path(self, tmp_path):
        from tools.paper_parse import _is_relative_to
        root = tmp_path / "root"
        other = tmp_path / "other" / "file.txt"
        assert _is_relative_to(other, root) is False

    def test_same_path(self, tmp_path):
        from tools.paper_parse import _is_relative_to
        root = tmp_path / "root"
        assert _is_relative_to(root, root) is True
