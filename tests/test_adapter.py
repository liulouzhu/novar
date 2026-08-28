"""tests/test_adapter.py — AgentAdapter 渠道 tool_context 隔离测试

验证 _build_tool_context 在有/无 user_id 时正确构造 tool_context，
并在 streaming / buffering 两条路径中传递给 agent.run_turn。
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novare.channels.adapter import AgentAdapter
from novare.channels.events import InboundMessage


def _make_msg(content: str = "hi", *, stream: bool = False) -> InboundMessage:
    """构造测试用 InboundMessage。"""
    return InboundMessage(
        channel="test_channel",
        sender_id="platform_user_1",
        chat_id="chat_1",
        content=content,
        metadata={"_wants_stream": stream} if stream else {},
    )


def _make_bus() -> MagicMock:
    """构造带 Async outbound queue 的 mock MessageBus。"""
    bus = MagicMock()
    bus.outbound = asyncio.Queue()
    return bus


@pytest.fixture
def agent_service(tmp_path):
    """构造最小 mock AgentService（无 DB 依赖）。"""
    svc = MagicMock()
    svc.agent = AsyncMock()
    svc.config = MagicMock()
    svc.config.system_prompt = "test prompt"
    svc._workspace_for = MagicMock(return_value=tmp_path / "ws")
    svc.load_session = AsyncMock(return_value=MagicMock(session_id="sess-1", messages=[]))
    return svc


# ── _build_tool_context 单元测试 ──────────────────────────────────────────────


class TestBuildToolContext:
    """_build_tool_context 在有/无 user_id 下的行为。"""

    def test_with_user_id_includes_user_id_and_workspace(self, tmp_path, agent_service):
        ws = tmp_path / "ws"
        adapter = AgentAdapter(bus=_make_bus(), agent_service=agent_service)
        ctx = adapter._build_tool_context("user-42")
        assert ctx is not None
        assert ctx["user_id"] == "user-42"
        assert ctx["workspace"] == str(ws)
        agent_service._workspace_for.assert_called_once_with("user-42")

    def test_without_user_id_returns_none(self, agent_service):
        adapter = AgentAdapter(bus=_make_bus(), agent_service=agent_service)
        ctx = adapter._build_tool_context(None)
        assert ctx is None

    def test_workspace_value_is_string(self, agent_service):
        """workspace 值必须是字符串（不能是 Path 对象）。"""
        adapter = AgentAdapter(bus=_make_bus(), agent_service=agent_service)
        ctx = adapter._build_tool_context("u-1")
        assert isinstance(ctx["workspace"], str)

    def test_fallback_to_get_user_workspace_when_no_method(self):
        """agent_service 没有 _workspace_for 时退回 get_user_workspace。"""
        svc = MagicMock(spec=[])  # 无任何属性
        adapter = AgentAdapter(bus=_make_bus(), agent_service=svc)
        with patch("novare.channels.adapter.get_user_workspace", return_value="/fallback/path") as mock_guw:
            ctx = adapter._build_tool_context("u-99")
        assert ctx["workspace"] == "/fallback/path"
        mock_guw.assert_called_once_with("u-99")

    def test_fallback_to_get_user_workspace_when_returns_none(self):
        """agent_service._workspace_for 返回 None 时退回 get_user_workspace。"""
        svc = MagicMock()
        svc._workspace_for.return_value = None
        adapter = AgentAdapter(bus=_make_bus(), agent_service=svc)
        with patch("novare.channels.adapter.get_user_workspace", return_value="/fb2") as mock_guw:
            ctx = adapter._build_tool_context("u-77")
        assert ctx["workspace"] == "/fb2"
        mock_guw.assert_called_once_with("u-77")


# ── 端到端：streaming 路径传递 tool_context ────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_passes_workspace_in_tool_context(tmp_path, agent_service):
    """streaming 路径：有 user_id 时 run_turn 收到含 workspace 的 tool_context。"""
    bus = _make_bus()
    adapter = AgentAdapter(bus=bus, agent_service=agent_service)
    with patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="user-1"):
        await adapter._handle_one(_make_msg("hello", stream=True))

    agent_service.agent.run_turn.assert_awaited_once()
    call_kwargs = agent_service.agent.run_turn.call_args
    ctx = call_kwargs.kwargs.get("tool_context")
    assert ctx is not None
    assert ctx["user_id"] == "user-1"
    assert ctx["workspace"] == str(tmp_path / "ws")
    assert isinstance(ctx["workspace"], str)


@pytest.mark.asyncio
async def test_buffering_passes_workspace_in_tool_context(tmp_path, agent_service):
    """buffering 路径：有 user_id 时 run_turn 收到含 workspace 的 tool_context。"""
    bus = _make_bus()
    adapter = AgentAdapter(bus=bus, agent_service=agent_service)
    with patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="user-1"):
        await adapter._handle_one(_make_msg("hi", stream=False))

    agent_service.agent.run_turn.assert_awaited_once()
    call_kwargs = agent_service.agent.run_turn.call_args
    ctx = call_kwargs.kwargs.get("tool_context")
    assert ctx is not None
    assert ctx["user_id"] == "user-1"
    assert ctx["workspace"] == str(tmp_path / "ws")
    assert isinstance(ctx["workspace"], str)


# ── 无 user_id 时保持兼容 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_no_user_id_tool_context_is_none():
    """无 user_id 时 streaming 路径 tool_context 应为 None。"""
    bus = _make_bus()
    svc = MagicMock()
    svc.agent = AsyncMock()
    svc.config = MagicMock()
    svc.config.system_prompt = "prompt"
    svc.load_session = AsyncMock(return_value=MagicMock(session_id="s", messages=[]))

    adapter = AgentAdapter(bus=bus, agent_service=svc)
    with patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value=None):
        await adapter._handle_one(_make_msg("x", stream=True))

    svc.agent.run_turn.assert_awaited_once()
    call_kwargs = svc.agent.run_turn.call_args
    ctx = call_kwargs.kwargs.get("tool_context")
    assert ctx is None


@pytest.mark.asyncio
async def test_buffering_no_user_id_tool_context_is_none():
    """无 user_id 时 buffering 路径 tool_context 应为 None。"""
    bus = _make_bus()
    svc = MagicMock()
    svc.agent = AsyncMock()
    svc.config = MagicMock()
    svc.config.system_prompt = "prompt"
    svc.load_session = AsyncMock(return_value=MagicMock(session_id="s", messages=[]))

    adapter = AgentAdapter(bus=bus, agent_service=svc)
    with patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value=None):
        await adapter._handle_one(_make_msg("x", stream=False))

    svc.agent.run_turn.assert_awaited_once()
    call_kwargs = svc.agent.run_turn.call_args
    ctx = call_kwargs.kwargs.get("tool_context")
    assert ctx is None


# ── _handle_one 使用 _resolve_user 的返回值 ───────────────────────────────────


@pytest.mark.asyncio
async def test_handle_one_uses_resolved_user_id(tmp_path, agent_service):
    """_handle_one 调用 _resolve_user 并将其返回值传给 _build_tool_context。"""
    bus = _make_bus()
    adapter = AgentAdapter(bus=bus, agent_service=agent_service)
    with patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="resolved-uid"):
        await adapter._handle_one(_make_msg("msg"))

    agent_service.agent.run_turn.assert_awaited_once()
    call_kwargs = agent_service.agent.run_turn.call_args
    ctx = call_kwargs.kwargs.get("tool_context")
    assert ctx["user_id"] == "resolved-uid"
    assert "workspace" in ctx
