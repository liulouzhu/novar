"""tests/test_subagent_workspace.py — 子智能体 workspace 继承测试

验证 handle_spawn_subagent 将父 agent 的 workspace 正确传递给 run_subagent，
确保子智能体的文件类工具使用用户隔离 workspace 而非全局 workspace。
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

from novare.subagents.registry import SubagentRegistry
from novare.subagents.types import SubagentType


def _make_kwargs(*, user_id=None, workspace=None):
    """构造 handle_spawn_subagent 的 kwargs。"""
    return {
        "subagent_registry": SubagentRegistry(),
        "parent_tool_registry": MagicMock(),
        "llm_client": MagicMock(),
        "system_prompt": "test",
        "default_max_iterations": 8,
        "turn_timeout": 60,
        **({"user_id": user_id} if user_id else {}),
        **({"workspace": workspace} if workspace else {}),
    }


class TestSubagentWorkspaceInheritance:
    """handle_spawn_subagent 必须将 workspace 传递给子智能体。"""

    @pytest.mark.asyncio
    async def test_workspace_passed_to_run_subagent(self):
        """有 user_id + workspace → run_subagent 收到两者。"""
        user_ws = Path("/workspace/users/u-abc")
        captured = {}

        async def fake_run_subagent(**kwargs):
            captured["tool_context"] = kwargs.get("tool_context")
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs(user_id="u-abc", workspace=user_ws)
            await handle_spawn_subagent(
                {"task": "read a file", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        assert captured["tool_context"] == {
            "user_id": "u-abc",
            "workspace": str(user_ws),
        }

    @pytest.mark.asyncio
    async def test_user_id_only_no_workspace(self):
        """有 user_id 但无 workspace → 只传 user_id。"""
        captured = {}

        async def fake_run_subagent(**kwargs):
            captured["tool_context"] = kwargs.get("tool_context")
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs(user_id="u-abc")
            await handle_spawn_subagent(
                {"task": "do something", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        assert captured["tool_context"] == {"user_id": "u-abc"}

    @pytest.mark.asyncio
    async def test_no_user_id_tool_context_is_none(self):
        """无 user_id → tool_context 为 None（CLI 模式）。"""
        captured = {}

        async def fake_run_subagent(**kwargs):
            captured["tool_context"] = kwargs.get("tool_context")
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs()
            await handle_spawn_subagent(
                {"task": "do something", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        assert captured["tool_context"] is None

    @pytest.mark.asyncio
    async def test_workspace_as_string(self):
        """workspace 为字符串时也正确传递。"""
        captured = {}

        async def fake_run_subagent(**kwargs):
            captured["tool_context"] = kwargs.get("tool_context")
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs(user_id="u-1", workspace="/some/path")
            await handle_spawn_subagent(
                {"task": "test", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        assert captured["tool_context"] == {"user_id": "u-1", "workspace": "/some/path"}

    @pytest.mark.asyncio
    async def test_workspace_str_conversion(self):
        """Path 对象的 workspace 被转为字符串。"""
        captured = {}

        async def fake_run_subagent(**kwargs):
            captured["tool_context"] = kwargs.get("tool_context")
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs(user_id="u-1", workspace=Path("/ws/user"))
            await handle_spawn_subagent(
                {"task": "test", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        tc = captured["tool_context"]
        assert tc["workspace"] == str(Path("/ws/user"))
        assert isinstance(tc["workspace"], str)
