"""五、WebSocket/API 恢复入口 route 级测试。

验证 chat route 从 send payload 读取可选字段 recovery_run_id：
- 非空、限长字符串时透传给 agent_service.run_turn(..., recovery_run_id=...)
- 非法值（非字符串 / 空 / 超长 / 无 user）→ fail closed，返回统一错误事件，不执行新 turn
- 不传时保持普通新 turn 行为
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fastapi.testclient import TestClient


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def close(self):
        pass


class _FakeSessionRepo:
    def __init__(self, db, user_uuid):
        pass

    async def get_by_id(self, session_id):
        # session 存在（属当前用户）
        return object()


class _FakeAgentService:
    """fake agent_service：记录 run_turn 调用，立即回 done 事件。"""

    def __init__(self):
        self.config = SimpleNamespace(turn_timeout=300)
        self.run_turn_calls = []

    async def load_session(self, session_id, user_id=None):
        session = MagicMock()
        session.session_id = session_id
        session.messages = []
        return session

    async def run_turn(self, session, user_input, queue, user_id=None, recovery_run_id=None):
        self.run_turn_calls.append({
            "user_input": user_input,
            "user_id": user_id,
            "recovery_run_id": recovery_run_id,
        })
        await queue.put({"type": "done"})


def _patch_deps(monkeypatch, fake_svc):
    import web.backend.routes.chat as chat_mod
    import web.backend.app as app_mod

    monkeypatch.setattr(app_mod, "agent_service", fake_svc)
    monkeypatch.setattr(chat_mod, "SessionRepository", _FakeSessionRepo)
    monkeypatch.setattr(chat_mod, "get_session_factory", lambda: lambda: _FakeDB())
    # decode_access_token 返回合法 UUID 字符串（SessionRepository 用 UUID(user_id_str)）
    fake_user_id = str(uuid.uuid4())
    monkeypatch.setattr(chat_mod, "decode_access_token", lambda token: fake_user_id)
    return chat_mod


def _ws_url(session_id: str) -> str:
    return f"/ws/chat/{session_id}?token=dummy-token"


def test_chat_route_passes_recovery_run_id(monkeypatch):
    from web.backend.app import app

    fake_svc = _FakeAgentService()
    _patch_deps(monkeypatch, fake_svc)

    session_id = f"s-{uuid.uuid4().hex[:8]}"
    client = TestClient(app)
    with client.websocket_connect(_ws_url(session_id)) as ws:
        ws.send_json({"type": "send", "content": "hi", "recovery_run_id": "run-123"})
        event = ws.receive_json()
        assert event["type"] == "done"

    assert len(fake_svc.run_turn_calls) == 1
    assert fake_svc.run_turn_calls[0]["recovery_run_id"] == "run-123"
    assert fake_svc.run_turn_calls[0]["user_input"] == "hi"


def test_chat_route_without_recovery_run_id_plain_turn(monkeypatch):
    from web.backend.app import app

    fake_svc = _FakeAgentService()
    _patch_deps(monkeypatch, fake_svc)

    session_id = f"s-{uuid.uuid4().hex[:8]}"
    client = TestClient(app)
    with client.websocket_connect(_ws_url(session_id)) as ws:
        ws.send_json({"type": "send", "content": "hello"})
        event = ws.receive_json()
        assert event["type"] == "done"

    assert len(fake_svc.run_turn_calls) == 1
    assert fake_svc.run_turn_calls[0]["recovery_run_id"] is None


@pytest.mark.parametrize("bad_value", [123, "", "   ", "x" * 129])
def test_chat_route_invalid_recovery_run_id_fails_closed(monkeypatch, bad_value):
    from web.backend.app import app

    fake_svc = _FakeAgentService()
    _patch_deps(monkeypatch, fake_svc)

    session_id = f"s-{uuid.uuid4().hex[:8]}"
    client = TestClient(app)
    with client.websocket_connect(_ws_url(session_id)) as ws:
        ws.send_json({"type": "send", "content": "hi", "recovery_run_id": bad_value})
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["code"] == "RECOVERY_RESUME_FAILED"
        assert "无法恢复" in event["message"]

    # 无效恢复请求 → 不执行新 turn
    assert fake_svc.run_turn_calls == []
