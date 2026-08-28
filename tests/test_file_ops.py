"""tests/test_file_ops.py"""

import os
import sys

import pytest

from novare.tools.file_ops import (
    handle_read_file,
    handle_write_file,
    handle_edit_file,
    handle_glob_search,
    handle_grep_search,
)


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_workspace):
        (tmp_workspace / "test.txt").write_text("Hello world", encoding="utf-8")
        result = await handle_read_file({"path": str(tmp_workspace / "test.txt")}, workspace=tmp_workspace)
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_workspace):
        result = await handle_read_file({"path": str(tmp_workspace / "nope.txt")}, workspace=tmp_workspace)
        assert "Error" in result or "error" in result.lower()


class TestWriteFile:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_workspace):
        path = tmp_workspace / "new.txt"
        result = await handle_write_file({"path": str(path), "content": "test data"}, workspace=tmp_workspace)
        assert "OK" in result or "written" in result.lower() or "success" in result.lower()
        assert path.read_text(encoding="utf-8") == "test data"

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_workspace):
        path = tmp_workspace / "existing.txt"
        path.write_text("old", encoding="utf-8")
        await handle_write_file({"path": str(path), "content": "new"}, workspace=tmp_workspace)
        assert path.read_text(encoding="utf-8") == "new"


class TestEditFile:
    @pytest.mark.asyncio
    async def test_edit_replaces_string(self, tmp_workspace):
        path = tmp_workspace / "edit.txt"
        path.write_text("Hello world", encoding="utf-8")
        result = await handle_edit_file({
            "path": str(path),
            "old_string": "world",
            "new_string": "Python",
        }, workspace=tmp_workspace)
        assert "OK" in result or "success" in result.lower() or "replaced" in result.lower()
        assert path.read_text(encoding="utf-8") == "Hello Python"

    @pytest.mark.asyncio
    async def test_edit_string_not_found(self, tmp_workspace):
        path = tmp_workspace / "edit.txt"
        path.write_text("Hello world", encoding="utf-8")
        result = await handle_edit_file({
            "path": str(path),
            "old_string": "xyz",
            "new_string": "abc",
        }, workspace=tmp_workspace)
        assert "Error" in result or "not found" in result.lower() or "not match" in result.lower()


class TestGlobSearch:
    @pytest.mark.asyncio
    async def test_glob_finds_files(self, tmp_workspace):
        (tmp_workspace / "a.py").write_text("x=1", encoding="utf-8")
        (tmp_workspace / "b.py").write_text("y=2", encoding="utf-8")
        (tmp_workspace / "c.txt").write_text("z=3", encoding="utf-8")
        result = await handle_glob_search({"pattern": "*.py", "path": str(tmp_workspace)}, workspace=tmp_workspace)
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result


class TestGrepSearch:
    @pytest.mark.asyncio
    async def test_grep_finds_content(self, tmp_workspace):
        (tmp_workspace / "f1.py").write_text("def hello():\n    pass", encoding="utf-8")
        (tmp_workspace / "f2.py").write_text("def world():\n    pass", encoding="utf-8")
        result = await handle_grep_search({"pattern": "hello", "path": str(tmp_workspace)}, workspace=tmp_workspace)
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_workspace):
        (tmp_workspace / "f1.py").write_text("def hello():\n    pass", encoding="utf-8")
        result = await handle_grep_search({"pattern": "zzzzz", "path": str(tmp_workspace)}, workspace=tmp_workspace)
        assert "no match" in result.lower() or "not found" in result.lower() or result.strip() == ""


# ── 路径安全测试 ──────────────────────────────────────────────────────────────

class TestPathSecurity:
    """验证所有 file_ops 工具拒绝逃逸 workspace 的路径。"""

    @pytest.mark.asyncio
    async def test_read_file_relative_escape_rejected(self, tmp_workspace):
        """read_file("../.env") 应被拒绝。"""
        result = await handle_read_file({"path": "../.env"}, workspace=tmp_workspace)
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_write_file_relative_escape_rejected(self, tmp_workspace):
        """write_file("../evil.txt") 应被拒绝。"""
        result = await handle_write_file(
            {"path": "../evil.txt", "content": "malicious"}, workspace=tmp_workspace,
        )
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_read_file_absolute_escape_rejected(self, tmp_workspace):
        """绝对路径指向 workspace 外应被拒绝。"""
        outside = tmp_workspace.parent / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        result = await handle_read_file({"path": str(outside)}, workspace=tmp_workspace)
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_write_file_absolute_escape_rejected(self, tmp_workspace):
        """绝对路径写入 workspace 外应被拒绝。"""
        outside = tmp_workspace.parent / "evil.txt"
        result = await handle_write_file(
            {"path": str(outside), "content": "malicious"}, workspace=tmp_workspace,
        )
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_edit_file_escape_rejected(self, tmp_workspace):
        """edit_file 对 workspace 外路径应被拒绝。"""
        outside = tmp_workspace.parent / "secret.txt"
        outside.write_text("old", encoding="utf-8")
        result = await handle_edit_file(
            {"path": str(outside), "old_string": "old", "new_string": "new"},
            workspace=tmp_workspace,
        )
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_glob_search_escape_rejected(self, tmp_workspace):
        """glob_search 的 path="../" 应被拒绝。"""
        result = await handle_glob_search(
            {"pattern": "*", "path": ".."}, workspace=tmp_workspace,
        )
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_grep_search_escape_rejected(self, tmp_workspace):
        """grep_search 的 path="../" 应被拒绝。"""
        result = await handle_grep_search(
            {"pattern": "secret", "path": "../"}, workspace=tmp_workspace,
        )
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_read_file_deep_relative_escape_rejected(self, tmp_workspace):
        """多层 ../ 逃逸应被拒绝。"""
        result = await handle_read_file({"path": "../../etc/passwd"}, workspace=tmp_workspace)
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result

    @pytest.mark.asyncio
    async def test_read_file_within_workspace_ok(self, tmp_workspace):
        """workspace 内的相对路径应正常工作。"""
        (tmp_workspace / "sub").mkdir()
        (tmp_workspace / "sub" / "ok.txt").write_text("safe", encoding="utf-8")
        result = await handle_read_file({"path": "sub/ok.txt"}, workspace=tmp_workspace)
        assert result == "safe"

    @pytest.mark.asyncio
    async def test_write_file_within_workspace_ok(self, tmp_workspace):
        """workspace 内的相对路径写入应正常工作。"""
        result = await handle_write_file(
            {"path": "sub/deep/file.txt", "content": "hello"}, workspace=tmp_workspace,
        )
        assert "OK" in result
        assert (tmp_workspace / "sub" / "deep" / "file.txt").read_text(encoding="utf-8") == "hello"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks require privileges on Windows")
    @pytest.mark.asyncio
    async def test_symlink_escape_rejected(self, tmp_workspace):
        """符号链接指向 workspace 外应被拒绝。"""
        outside = tmp_workspace.parent / "outside_secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = tmp_workspace / "link.txt"
        os.symlink(outside, link)
        result = await handle_read_file({"path": str(link)}, workspace=tmp_workspace)
        assert "Error" in result
        assert "outside workspace" in result.lower() or "Path outside" in result
