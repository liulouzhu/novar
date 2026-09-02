"""tests/test_registry.py"""

import json
from unittest.mock import AsyncMock

import pytest

from novare.subagents.tool_executor import SubagentToolExecutor
from novare.recovery.policy import RetryPolicy
from novare.tools.registry import ToolRegistry, ToolDef
from novare.tools.reviewer_evaluate import handle_reviewer_evaluate
from novare.llm_client import LLMResponse


class TestToolDef:
    def test_to_openai_tool(self):
        tool = ToolDef(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=None,
        )
        openai_tool = tool.to_openai_tool()
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "read_file"
        assert openai_tool["function"]["description"] == "Read a file"

    def test_idempotency_inferred_for_known_read_tool(self):
        """未指定 idempotency 的已知只读工具推断为 read。"""
        tool = ToolDef(name="read_file", description="Read", parameters={}, handler=None)
        assert tool.idempotency == "read"
        assert tool.retry_policy is not None
        assert tool.retry_policy.max_attempts == 3

    def test_explicit_non_idempotent_not_overridden(self):
        """显式 non_idempotent 的已知名称工具绝不能被 __post_init__ 改成 read。"""
        tool = ToolDef(
            name="read_file", description="Read", parameters={}, handler=None,
            idempotency="non_idempotent",
        )
        assert tool.idempotency == "non_idempotent"

    def test_explicit_read_preserved(self):
        tool = ToolDef(name="x", description="d", parameters={}, handler=None, idempotency="read")
        assert tool.idempotency == "read"

    def test_invalid_idempotency_falls_back_safe(self):
        """非法 idempotency 值安全回退到 non_idempotent。"""
        tool = ToolDef(name="x", description="d", parameters={}, handler=None, idempotency="maybe")
        assert tool.idempotency == "non_idempotent"

    def test_unknown_tool_defaults_conservative(self):
        """未知工具（名字不在只读集合）默认 non_idempotent、不重试。"""
        tool = ToolDef(name="code_execute", description="d", parameters={}, handler=None)
        assert tool.idempotency == "non_idempotent"
        assert tool.retry_policy is None

    def test_mcp_tool_not_inferred_by_name(self):
        """MCP 工具不按名字推断：未显式声明时默认 non_idempotent、不重试。"""
        tool = ToolDef(
            name="paper_search", description="d", parameters={}, handler=None,
            source="mcp:research",
        )
        assert tool.idempotency == "non_idempotent"
        assert tool.retry_policy is None

    def test_mcp_tool_explicit_declaration_enables_retry(self):
        """MCP 工具显式声明 idempotency/retry_policy 后仍可安全重试。"""
        tool = ToolDef(
            name="paper_search", description="d", parameters={}, handler=None,
            source="mcp:research",
            idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
        )
        assert tool.idempotency == "read"
        assert tool.retry_policy is not None
        assert tool.retry_policy.max_attempts == 3

    def test_mcp_other_tool_defaults_conservative(self):
        """其余 MCP 工具默认 non_idempotent、max_attempts=1。"""
        tool = ToolDef(
            name="knowledge_graph", description="d", parameters={}, handler=None,
            source="mcp:research",
        )
        assert tool.idempotency == "non_idempotent"
        assert tool.retry_policy is None


class TestSubagentExecutorDelegation:
    def test_idempotency_delegated_to_parent(self, tmp_workspace):
        """SubagentToolExecutor 委托父注册表的 idempotency_for。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        registry.register_tool(ToolDef(
            name="read_file", description="r", parameters={},
            handler=None, idempotency="read",
        ))
        executor = SubagentToolExecutor(registry, {"read_file"})
        assert executor.idempotency_for("read_file") == "read"
        assert executor.retry_policy_for("read_file") is not None

    def test_idempotency_outside_allowlist_conservative(self, tmp_workspace):
        """白名单外工具保守返回 non_idempotent。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        registry.register_tool(ToolDef(
            name="reader", description="r", parameters={},
            handler=None, idempotency="read",
        ))
        executor = SubagentToolExecutor(registry, {"other"})
        assert executor.idempotency_for("reader") == "non_idempotent"


class TestToolRegistry:
    def test_register_builtin_tools(self, tmp_workspace):
        registry = ToolRegistry(workspace=tmp_workspace)
        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "glob_search" in names
        assert "grep_search" in names

    @pytest.mark.asyncio
    async def test_execute_builtin_tool(self, tmp_workspace):
        (tmp_workspace / "test.txt").write_text("hello", encoding="utf-8")
        registry = ToolRegistry(workspace=tmp_workspace)
        result = await registry.execute("read_file", {"path": str(tmp_workspace / "test.txt")})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tmp_workspace):
        """重构后：未知工具返回结构化错误 JSON（classifier 可直接分类）。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        result = await registry.execute("nonexistent_tool", {})
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert parsed["error_code"] == "UNKNOWN_TOOL"
        assert parsed["retryable"] is False
        assert "nonexistent_tool" in parsed["error"]

    def test_to_openai_tools(self, tmp_workspace):
        registry = ToolRegistry(workspace=tmp_workspace)
        tools = registry.to_openai_tools()
        assert len(tools) >= 5
        assert all(t["type"] == "function" for t in tools)

    def test_register_mcp_tools(self, tmp_workspace):
        registry = ToolRegistry(workspace=tmp_workspace)
        registry.register_tool(ToolDef(
            name="paper_search",
            description="Search papers",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=None,
        ))
        tools = registry.list_tools()
        names = [t.name for t in tools]
        assert "paper_search" in names


def _make_fake_reviewer_llm(response_content: str = '{"reviews": []}'):
    """构造一个 mock LLMClient，collect_stream 返回指定内容。"""
    fake = AsyncMock()
    fake.collect_stream.return_value = LLMResponse(
        content=response_content,
        tool_calls=[],
        stop_reason="stop",
    )
    return fake


class TestReviewerEvaluate:
    """验证 reviewer_evaluate 能正确拿到 reviewer_llm。"""

    @pytest.mark.asyncio
    async def test_registry_execute_passes_reviewer_llm(self, tmp_workspace):
        """通过 registry.execute + tool_context 传入 reviewer_llm 应该生效。"""
        fake_llm = _make_fake_reviewer_llm()
        registry = ToolRegistry(workspace=tmp_workspace)
        args = {
            "topic": "test topic",
            "stage": "candidates",
            "candidates": [{"title": "idea", "problem": "p", "idea": "i"}],
        }
        result = await registry.execute(
            "reviewer_evaluate", args,
            tool_context={"reviewer_llm": fake_llm},
        )
        fake_llm.collect_stream.assert_awaited_once()
        assert "评审模型未配置" not in result

    @pytest.mark.asyncio
    async def test_handle_direct_kwargs(self, tmp_workspace):
        """直接调用 handler，reviewer_llm 作为顶层 kwargs 传入。"""
        fake_llm = _make_fake_reviewer_llm()
        args = {
            "topic": "test",
            "stage": "candidates",
            "candidates": [{"title": "t", "problem": "p", "idea": "i"}],
        }
        result = await handle_reviewer_evaluate(args, reviewer_llm=fake_llm)
        fake_llm.collect_stream.assert_awaited_once()
        assert "评审模型未配置" not in result

    @pytest.mark.asyncio
    async def test_handle_legacy_tool_context_kwarg(self, tmp_workspace):
        """兼容旧格式：kwargs["tool_context"]["reviewer_llm"]。"""
        fake_llm = _make_fake_reviewer_llm()
        args = {
            "topic": "test",
            "stage": "candidates",
            "candidates": [{"title": "t", "problem": "p", "idea": "i"}],
        }
        result = await handle_reviewer_evaluate(args, tool_context={"reviewer_llm": fake_llm})
        fake_llm.collect_stream.assert_awaited_once()
        assert "评审模型未配置" not in result

    @pytest.mark.asyncio
    async def test_no_reviewer_llm_returns_error(self, tmp_workspace):
        """未传 reviewer_llm 时应返回配置提示。"""
        args = {
            "topic": "test",
            "stage": "candidates",
            "candidates": [{"title": "t", "problem": "p", "idea": "i"}],
        }
        result = await handle_reviewer_evaluate(args)
        assert "评审模型未配置" in result


class TestDefaultToolContext:
    """C1: set_default_tool_context 替代 monkey patch"""

    def test_set_default_tool_context_stores_dict(self, tmp_workspace):
        """set_default_tool_context 存储上下文。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        ctx = {"foo": "bar"}
        registry.set_default_tool_context(ctx)
        assert registry._default_tool_context == {"foo": "bar"}
        # 修改原 dict 不影响已存储的副本
        ctx["baz"] = 1
        assert "baz" not in registry._default_tool_context

    def test_set_default_tool_context_none_clears(self, tmp_workspace):
        """传 None 清空默认上下文。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        registry.set_default_tool_context({"a": 1})
        registry.set_default_tool_context(None)
        assert registry._default_tool_context == {}

    @pytest.mark.asyncio
    async def test_execute_merges_default_and_call_context(self, tmp_workspace):
        """execute 合并默认上下文和调用时上下文，调用时优先。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        captured_kwargs = {}

        async def spy_handler(args, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        registry.register_tool(ToolDef(
            name="spy_tool",
            description="spy",
            parameters={"type": "object", "properties": {}},
            handler=spy_handler,
            source="builtin:context",
        ))

        registry.set_default_tool_context({"alpha": 1, "beta": 2})
        await registry.execute("spy_tool", {}, tool_context={"beta": 99, "gamma": 3})

        assert captured_kwargs["alpha"] == 1
        assert captured_kwargs["beta"] == 99   # 调用时覆盖默认
        assert captured_kwargs["gamma"] == 3
        assert captured_kwargs["workspace"] == tmp_workspace

    @pytest.mark.asyncio
    async def test_execute_no_context_for_plain_builtin(self, tmp_workspace):
        """普通 builtin 工具不注入默认上下文。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        captured_kwargs = {}

        async def spy_handler(args, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok"

        registry.register_tool(ToolDef(
            name="spy_plain",
            description="spy",
            parameters={"type": "object", "properties": {}},
            handler=spy_handler,
            source="builtin",  # 普通 builtin，不是 builtin:context
        ))

        registry.set_default_tool_context({"secret": "value"})
        await registry.execute("spy_plain", {})

        assert "secret" not in captured_kwargs
        assert captured_kwargs["workspace"] == tmp_workspace

    @pytest.mark.asyncio
    async def test_reviewer_evaluate_gets_default_context(self, tmp_workspace):
        """reviewer_evaluate 通过默认上下文也能拿到 reviewer_llm。"""
        registry = ToolRegistry(workspace=tmp_workspace)
        fake_llm = _make_fake_reviewer_llm()

        # 只设默认上下文，不传 tool_context
        registry.set_default_tool_context({"reviewer_llm": fake_llm})

        args = {
            "topic": "test",
            "stage": "candidates",
            "candidates": [{"title": "t", "problem": "p", "idea": "i"}],
        }
        result = await registry.execute("reviewer_evaluate", args)
        fake_llm.collect_stream.assert_awaited_once()
        assert "评审模型未配置" not in result

    def test_register_subagent_tools_no_monkey_patch(self, tmp_workspace, monkeypatch):
        """register_subagent_tools 后 execute 仍是原方法（非 monkey patch）。"""
        # 导入链需要 DATABASE_URL（session.py → db → base.py）
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
        from novare.subagents.registry import SubagentRegistry
        from novare.subagents.tools import register_subagent_tools
        from unittest.mock import MagicMock

        registry = ToolRegistry(workspace=tmp_workspace)
        # 记录原始的底层函数
        original_func = registry.execute.__func__

        sub_reg = SubagentRegistry()
        fake_llm = MagicMock()
        register_subagent_tools(
            tool_registry=registry,
            subagent_registry=sub_reg,
            llm_client=fake_llm,
            system_prompt="test",
        )

        # execute 的底层函数不应改变（不是闭包替换）
        assert registry.execute.__func__ is original_func
        # 默认上下文已被设置
        assert registry._default_tool_context.get("subagent_registry") is sub_reg
        # 默认上下文不含 workspace：子智能体 workspace 必须来自 per-user
        # tool_context，防止继承多用户共享的全局目录
        assert "workspace" not in registry._default_tool_context


class TestFileToolWorkspaceOverride:
    """Web 多用户隔离：文件类 builtin 工具应使用 tool_context["workspace"]。"""

    @pytest.mark.asyncio
    async def test_write_file_uses_user_workspace(self, tmp_path):
        """write_file + tool_context.workspace → 写入用户目录，非全局目录。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "write_file",
            {"path": "hello.txt", "content": "from user"},
            tool_context={"workspace": str(user_ws)},
        )
        assert "8" in result or "from user" not in result  # 返回字符数
        assert (user_ws / "hello.txt").read_text(encoding="utf-8") == "from user"
        assert not (global_ws / "hello.txt").exists()

    @pytest.mark.asyncio
    async def test_read_file_uses_user_workspace(self, tmp_path):
        """read_file + tool_context.workspace → 从用户目录读取。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()
        (user_ws / "data.txt").write_text("user data", encoding="utf-8")

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "read_file",
            {"path": "data.txt"},
            tool_context={"workspace": str(user_ws)},
        )
        assert "user data" in result

    @pytest.mark.asyncio
    async def test_edit_file_uses_user_workspace(self, tmp_path):
        """edit_file + tool_context.workspace → 编辑用户目录中的文件。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()
        (user_ws / "edit.txt").write_text("old text", encoding="utf-8")

        registry = ToolRegistry(workspace=global_ws)
        await registry.execute(
            "edit_file",
            {"path": "edit.txt", "old_string": "old text", "new_string": "new text"},
            tool_context={"workspace": str(user_ws)},
        )
        assert (user_ws / "edit.txt").read_text(encoding="utf-8") == "new text"

    @pytest.mark.asyncio
    async def test_glob_search_uses_user_workspace(self, tmp_path):
        """glob_search + tool_context.workspace → 在用户目录中搜索。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()
        (user_ws / "a.py").write_text("# python", encoding="utf-8")

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "glob_search",
            {"pattern": "*.py"},
            tool_context={"workspace": str(user_ws)},
        )
        assert "a.py" in result

    @pytest.mark.asyncio
    async def test_grep_search_uses_user_workspace(self, tmp_path):
        """grep_search + tool_context.workspace → 在用户目录中 grep。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()
        (user_ws / "code.py").write_text("hello_world = 42", encoding="utf-8")

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "grep_search",
            {"pattern": "hello_world"},
            tool_context={"workspace": str(user_ws)},
        )
        assert "hello_world" in result
        assert "code.py" in result

    @pytest.mark.asyncio
    async def test_no_tool_context_uses_global_workspace(self, tmp_path):
        """不传 tool_context 时，write_file 使用全局 workspace（CLI 模式）。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "write_file",
            {"path": "cli.txt", "content": "cli data"},
        )
        assert (global_ws / "cli.txt").read_text(encoding="utf-8") == "cli data"

    @pytest.mark.asyncio
    async def test_tool_context_without_workspace_uses_global(self, tmp_path):
        """tool_context 有其他 key 但无 workspace 时，仍用全局 workspace。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws)
        await registry.execute(
            "write_file",
            {"path": "fallback.txt", "content": "ok"},
            tool_context={"user_id": "u-123"},
        )
        assert (global_ws / "fallback.txt").read_text(encoding="utf-8") == "ok"

    @pytest.mark.asyncio
    async def test_mcp_handler_still_receives_user_id(self, tmp_path):
        """MCP 工具仍从 tool_context 接收 user_id，workspace 不影响 MCP。"""
        captured = {}

        async def fake_mcp_handler(args, **kwargs):
            captured.update(kwargs)
            return "mcp ok"

        registry = ToolRegistry(workspace=tmp_path)
        registry.register_tool(ToolDef(
            name="fake_mcp",
            description="Fake MCP tool",
            parameters={"type": "object", "properties": {}},
            handler=fake_mcp_handler,
            source="mcp:test",
        ))
        user_ws = tmp_path / "user_ws"
        user_ws.mkdir()
        await registry.execute(
            "fake_mcp",
            {},
            tool_context={"user_id": "u-abc", "workspace": str(user_ws)},
        )
        assert captured.get("user_id") == "u-abc"


class TestFileToolMultiUserFailClosed:
    """多用户模式下，文件类工具缺少 per-user workspace 时必须 fail-closed。"""

    @pytest.mark.asyncio
    async def test_missing_workspace_denied(self, tmp_path):
        """multi_user 注册表 + 无 tool_context → PERMISSION_DENIED，不落盘。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws, multi_user=True)
        result = await registry.execute(
            "read_file", {"path": "secret.txt"},
        )
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert parsed["error_code"] == "PERMISSION_DENIED"
        assert parsed["retryable"] is False
        assert not (global_ws / "secret.txt").exists()

    @pytest.mark.asyncio
    async def test_tool_context_without_workspace_denied(self, tmp_path):
        """tool_context 有 user_id 但无 workspace → 拒绝（不回退到共享目录）。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws, multi_user=True)
        result = await registry.execute(
            "write_file",
            {"path": "fallback.txt", "content": "ok"},
            tool_context={"user_id": "u-123"},
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "PERMISSION_DENIED"
        assert not (global_ws / "fallback.txt").exists()

    @pytest.mark.asyncio
    async def test_empty_workspace_denied(self, tmp_path):
        """workspace 为空字符串视为缺失，拒绝执行。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws, multi_user=True)
        result = await registry.execute(
            "read_file", {"path": "x.txt"},
            tool_context={"user_id": "u-1", "workspace": ""},
        )
        parsed = json.loads(result)
        assert parsed["error_code"] == "PERMISSION_DENIED"

    @pytest.mark.asyncio
    async def test_with_user_workspace_allowed(self, tmp_path):
        """multi_user 注册表 + per-user workspace → 正常执行。"""
        global_ws = tmp_path / "global"
        user_ws = tmp_path / "user"
        global_ws.mkdir()
        user_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws, multi_user=True)
        result = await registry.execute(
            "write_file",
            {"path": "hello.txt", "content": "user data"},
            tool_context={"user_id": "u-1", "workspace": str(user_ws)},
        )
        assert "Error" not in result
        assert (user_ws / "hello.txt").read_text(encoding="utf-8") == "user data"

    @pytest.mark.asyncio
    async def test_all_file_tools_fail_closed(self, tmp_path):
        """全部文件类工具在多用户模式下缺 workspace 时均拒绝。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws, multi_user=True)
        for name, args in (
            ("read_file", {"path": "a.txt"}),
            ("edit_file", {"path": "a.txt", "old_string": "x", "new_string": "y"}),
            ("glob_search", {"pattern": "*"}),
            ("grep_search", {"pattern": "x"}),
        ):
            result = await registry.execute(name, args)
            parsed = json.loads(result)
            assert parsed["error_code"] == "PERMISSION_DENIED", f"{name} 应拒绝执行"

    @pytest.mark.asyncio
    async def test_single_user_mode_unchanged(self, tmp_path):
        """单用户（默认）模式保持原行为：无 tool_context 用全局 workspace。"""
        global_ws = tmp_path / "global"
        global_ws.mkdir()

        registry = ToolRegistry(workspace=global_ws)
        result = await registry.execute(
            "write_file", {"path": "cli.txt", "content": "cli data"},
        )
        assert "Error" not in result
        assert (global_ws / "cli.txt").read_text(encoding="utf-8") == "cli data"
