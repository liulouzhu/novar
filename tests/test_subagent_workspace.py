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


class TestSubagentOwnershipHandlers:
    """多用户隔离：spawn 绑定创建者，check/list 按归属过滤。"""

    @pytest.mark.asyncio
    async def test_spawn_binds_user_id_to_record(self):
        """Web 模式 spawn 创建的记录绑定 user_id。"""
        reg = SubagentRegistry()

        async def fake_run_subagent(**kwargs):
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs(user_id="u-1", workspace="/ws/u-1")
            kwargs["subagent_registry"] = reg
            result = await handle_spawn_subagent(
                {"task": "test", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        output = json.loads(result)
        record = reg.get_owned(output["subagent_id"], "u-1")
        assert record is not None
        assert record.user_id == "u-1"

    @pytest.mark.asyncio
    async def test_spawn_multi_user_requires_user_id(self):
        """multi_user 模式下无 user_id → 拒绝创建，不产生无归属记录。"""
        reg = SubagentRegistry()
        from novare.subagents.tools import handle_spawn_subagent
        kwargs = _make_kwargs()
        kwargs["subagent_registry"] = reg
        kwargs["multi_user"] = True

        result = await handle_spawn_subagent(
            {"task": "test", "subagent_type": "general"},
            **kwargs,
        )

        output = json.loads(result)
        assert "error" in output
        assert "用户上下文" in output["error"]
        assert len(reg.list_all()) == 0

    @pytest.mark.asyncio
    async def test_spawn_cli_without_user_id_still_works(self):
        """CLI 模式（multi_user 缺省）无 user_id 仍可创建（原行为保持）。"""
        reg = SubagentRegistry()

        async def fake_run_subagent(**kwargs):
            return "done"

        with patch("novare.subagents.tools.run_subagent", side_effect=fake_run_subagent):
            from novare.subagents.tools import handle_spawn_subagent
            kwargs = _make_kwargs()
            kwargs["subagent_registry"] = reg
            await handle_spawn_subagent(
                {"task": "test", "subagent_type": "general", "await_result": True},
                **kwargs,
            )

        assert len(reg.list_all()) == 1
        assert reg.list_all()[0].user_id is None

    @pytest.mark.asyncio
    async def test_check_subagent_owner_can_read(self):
        """创建者本人可读取子智能体输出。"""
        from novare.subagents.tools import handle_check_subagent
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "my task", user_id="u-1")
        reg.complete(r.subagent_id, "my result")

        result = await handle_check_subagent(
            {"subagent_id": r.subagent_id},
            subagent_registry=reg, user_id="u-1",
        )
        output = json.loads(result)
        assert output["result"] == "my result"

    @pytest.mark.asyncio
    async def test_check_subagent_other_user_gets_not_found(self):
        """其他用户查询 → 与不存在相同的错误，不泄漏记录是否存在。"""
        from novare.subagents.tools import handle_check_subagent
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "secret task", user_id="u-1")
        reg.complete(r.subagent_id, "secret result")

        result = await handle_check_subagent(
            {"subagent_id": r.subagent_id},
            subagent_registry=reg, user_id="u-2",
        )
        output = json.loads(result)
        assert "error" in output
        assert f"未找到子智能体: {r.subagent_id}" == output["error"]
        assert "secret" not in result

    @pytest.mark.asyncio
    async def test_check_subagent_anonymous_cannot_read_owned(self):
        """无 user_id 的调用方查不到有归属的记录。"""
        from novare.subagents.tools import handle_check_subagent
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "t", user_id="u-1")
        reg.complete(r.subagent_id, "result")

        result = await handle_check_subagent(
            {"subagent_id": r.subagent_id},
            subagent_registry=reg,
        )
        output = json.loads(result)
        assert "error" in output

    @pytest.mark.asyncio
    async def test_list_subagents_filters_by_user(self):
        """list_subagents 只返回当前用户的子智能体。"""
        from novare.subagents.tools import handle_list_subagents
        reg = SubagentRegistry()
        reg.create(SubagentType.GENERAL, "alice task", user_id="u-1")
        other = reg.create(SubagentType.GENERAL, "bob task", user_id="u-2")

        result = await handle_list_subagents(
            {}, subagent_registry=reg, user_id="u-1",
        )
        output = json.loads(result)
        ids = [s["subagent_id"] for s in output["subagents"]]
        assert other.subagent_id not in ids
        assert all("bob task" not in s["task"] for s in output["subagents"])

    @pytest.mark.asyncio
    async def test_list_subagents_anonymous_sees_only_unowned(self):
        """无 user_id（CLI）只见无归属记录，看不到 Web 用户的。"""
        from novare.subagents.tools import handle_list_subagents
        reg = SubagentRegistry()
        reg.create(SubagentType.GENERAL, "web task", user_id="u-1")
        cli_record = reg.create(SubagentType.GENERAL, "cli task")

        result = await handle_list_subagents({}, subagent_registry=reg)
        output = json.loads(result)
        ids = [s["subagent_id"] for s in output["subagents"]]
        assert ids == [cli_record.subagent_id]
