"""tests/test_redis.py — Redis 功能测试

覆盖：
1. NovareConfig 读取 Redis 环境变量
2. RedisService disabled / 异常时安全降级（set_nx 返回 None）
3. RedisService JSON 操作 (set_json / get_json / expire)
4. AgentService.run_turn 并发锁
5. AgentService 任务状态写入
6. AgentService 协作式取消
7. AgentAdapter 渠道消息去重
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novare.channels.adapter import AgentAdapter
from novare.channels.events import InboundMessage


# ── 辅助构造 ─────────────────────────────────────────────────────────────────


def _make_msg(
    content: str = "hi",
    *,
    channel: str = "test_channel",
    sender_id: str = "user_1",
    chat_id: str = "chat_1",
    metadata: dict | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel=channel,
        sender_id=sender_id,
        chat_id=chat_id,
        content=content,
        metadata=metadata or {},
    )


def _make_bus() -> MagicMock:
    import asyncio as _asyncio
    bus = MagicMock()
    bus.outbound = _asyncio.Queue()
    return bus


def _make_agent_service(tmp_path) -> MagicMock:
    svc = MagicMock()
    svc.agent = AsyncMock()
    svc.config = MagicMock()
    svc.config.system_prompt = "test prompt"
    svc.config.turn_timeout = 300
    svc._workspace_for = MagicMock(return_value=tmp_path / "ws")
    svc.load_session = AsyncMock(return_value=MagicMock(session_id="sess-1", messages=[]))
    return svc


# ══════════════════════════════════════════════════════════════════════════════
# 1. NovareConfig — Redis 环境变量
# ══════════════════════════════════════════════════════════════════════════════


class TestNovareConfigRedis:
    """验证 NovareConfig.load() 正确读取 Redis 环境变量。"""

    def _load_cfg(self):
        """调用 NovareConfig.load() 并返回 cfg。"""
        from novare.config import NovareConfig
        return NovareConfig.load()

    def test_default_redis_disabled(self, monkeypatch):
        """默认值：redis_enabled=False, redis_url 有默认值。"""
        monkeypatch.delenv("NOVARE_REDIS_ENABLED", raising=False)
        monkeypatch.delenv("NOVARE_REDIS_URL", raising=False)
        cfg = self._load_cfg()
        assert cfg.redis_enabled is False
        assert cfg.redis_url == "redis://localhost:6379/0"

    @pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes", "Yes"])
    def test_env_enable_true_values(self, monkeypatch, val):
        """NOVARE_REDIS_ENABLED 的各种 true 值 → cfg.redis_enabled is True。"""
        monkeypatch.setenv("NOVARE_REDIS_ENABLED", val)
        monkeypatch.delenv("NOVARE_REDIS_URL", raising=False)
        cfg = self._load_cfg()
        assert cfg.redis_enabled is True

    @pytest.mark.parametrize("val", ["0", "false", "False", "no", ""])
    def test_env_enable_false_values(self, monkeypatch, val):
        """NOVARE_REDIS_ENABLED 的 false 值 → cfg.redis_enabled is False。"""
        monkeypatch.setenv("NOVARE_REDIS_ENABLED", val)
        monkeypatch.delenv("NOVARE_REDIS_URL", raising=False)
        cfg = self._load_cfg()
        assert cfg.redis_enabled is False

    def test_env_redis_url_override(self, monkeypatch):
        """NOVARE_REDIS_URL 覆盖默认值。"""
        monkeypatch.setenv("NOVARE_REDIS_URL", "redis://custom:6380/1")
        monkeypatch.delenv("NOVARE_REDIS_ENABLED", raising=False)
        cfg = self._load_cfg()
        assert cfg.redis_url == "redis://custom:6380/1"


# ══════════════════════════════════════════════════════════════════════════════
# 2. RedisService — disabled / 异常时安全降级
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisServiceDegradation:
    """Redis disabled / 异常时 set_nx 返回 None（降级态）。"""

    @pytest.fixture
    def svc(self):
        from web.backend.redis_service import RedisService
        return RedisService()

    @pytest.mark.asyncio
    async def test_initialize_disabled(self, svc):
        """enabled=False 时不创建连接。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        assert svc.is_available is False

    @pytest.mark.asyncio
    async def test_set_nx_disabled_returns_none(self, svc):
        """disabled 时 set_nx 返回 None（降级态）。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        result = await svc.set_nx("key", "val", 60)
        assert result is None

    @pytest.mark.asyncio
    async def test_set_nx_exception_returns_none(self, svc):
        """set_nx 操作异常时返回 None（降级态），并标记 unavailable。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.set = AsyncMock(side_effect=ConnectionError("broken pipe"))
            mock_from_url.return_value = mock_client

            await svc.initialize(enabled=True, url="redis://localhost:6379/0")
            assert svc.is_available is True

            result = await svc.set_nx("k", "v", 10)
            assert result is None
            assert svc.is_available is False

    @pytest.mark.asyncio
    async def test_get_disabled_returns_none(self, svc):
        """disabled 时 get 返回 None。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        result = await svc.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_disabled_no_error(self, svc):
        """disabled 时 delete 不抛异常。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        await svc.delete("key")  # 不应抛异常

    @pytest.mark.asyncio
    async def test_ping_disabled_returns_false(self, svc):
        """disabled 时 ping 返回 False。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        result = await svc.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_if_value_disabled_returns_false(self, svc):
        """disabled 时 delete_if_value 返回 False。"""
        await svc.initialize(enabled=False, url="redis://localhost:6379/0")
        result = await svc.delete_if_value("key", "val")
        assert result is False

    @pytest.mark.asyncio
    async def test_close_no_error_when_never_connected(self, svc):
        """从未连接时 close 不抛异常。"""
        await svc.close()

    @pytest.mark.asyncio
    async def test_connection_failure_sets_unavailable(self, svc):
        """连接失败后 is_available=False，set_nx 返回 None。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(side_effect=ConnectionError("refused"))
            mock_from_url.return_value = mock_client

            await svc.initialize(enabled=True, url="redis://bad:6379/0")
            assert svc.is_available is False

            # 后续调用仍安全降级
            assert await svc.set_nx("k", "v", 10) is None
            assert await svc.get("k") is None
            await svc.delete("k")
            assert await svc.ping() is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. AgentService.run_turn — 并发锁
# ══════════════════════════════════════════════════════════════════════════════


class TestRunTurnConcurrencyLock:
    """run_turn 的 Redis 并发锁行为。"""

    @pytest.fixture(autouse=True)
    def _set_db_url(self, monkeypatch):
        """确保 import agent_service 时 DATABASE_URL 存在。"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    @pytest.mark.asyncio
    async def test_lock_conflict_sends_error_and_done(self, tmp_path):
        """Redis 正常可用且 key 已存在（False），返回 error + done，不调用 agent。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-lock", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=False)

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        assert result == ""
        events = []
        while not queue.empty():
            events.append(await queue.get())
        assert events[0]["type"] == "error"
        assert "已有任务" in events[0]["message"]
        assert events[1]["type"] == "done"
        svc.agent.run_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_nx_degraded_continues_agent(self, tmp_path):
        """set_nx 返回 None（降级）时，不拒绝请求，继续执行 agent。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-degrade", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=None)
            mock_redis.delete = AsyncMock()
            mock_redis.set_json = AsyncMock(return_value=True)
            mock_redis.get = AsyncMock(return_value=None)

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        # agent 正常执行
        svc.agent.run_turn.assert_awaited_once()
        # queue 中没有 "已有任务" error
        events = []
        while not queue.empty():
            events.append(await queue.get())
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 0

    @pytest.mark.asyncio
    async def test_set_nx_degraded_does_not_release_lock(self, tmp_path):
        """set_nx 返回 None（降级）时，finally 不调用 delete_if_value。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-no-release", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=None)
            mock_redis.delete = AsyncMock()
            mock_redis.set_json = AsyncMock(return_value=True)
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        # 未获得锁，不应释放
        mock_redis.delete_if_value.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lock_success_releases_in_finally(self, tmp_path):
        """锁获取成功后，finally 中释放锁。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-ok", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.set_json = AsyncMock(return_value=True)
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        # 锁被释放（compare-and-delete）
        mock_redis.delete_if_value.assert_awaited_once()
        call_args = mock_redis.delete_if_value.call_args
        assert "lock:user:u-1:session:sess-ok" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_no_lock_when_redis_unavailable(self, tmp_path):
        """Redis 不可用时，不获取锁，保持原行为。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-noredis", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = False

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        # set_nx 不应被调用
        mock_redis.set_nx.assert_not_called()
        # agent 正常执行
        svc.agent.run_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_lock_when_no_user_id(self, tmp_path):
        """无 user_id 时不获取锁。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-anon", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True

            result = await svc.run_turn(session, "hello", queue, user_id=None)

        mock_redis.set_nx.assert_not_called()
        svc.agent.run_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lock_released_on_agent_error(self, tmp_path):
        """agent 执行异常时，锁仍被释放。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(side_effect=RuntimeError("boom"))

        session = MagicMock(session_id="sess-err", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.set_json = AsyncMock(return_value=True)
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        assert result == ""
        mock_redis.delete_if_value.assert_awaited_once()


# ══════════════════════════════════════════════════════════════════════════════
# 4. AgentAdapter — 渠道消息去重
# ══════════════════════════════════════════════════════════════════════════════


class TestChannelDedup:
    """渠道消息 Redis 去重行为。"""

    @pytest.mark.asyncio
    async def test_first_message_continues_processing(self, tmp_path):
        """首次消息：set_nx 返回 True，继续调用 agent。"""
        bus = _make_bus()
        agent_svc = _make_agent_service(tmp_path)
        adapter = AgentAdapter(bus=bus, agent_service=agent_svc)

        with patch("novare.channels.adapter.redis_service") as mock_redis, \
             patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="u-1"):
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)

            msg = _make_msg(metadata={"message_id": "msg-001"})
            await adapter._handle_one(msg)

        agent_svc.agent.run_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_message_skipped(self, tmp_path):
        """重复消息：set_nx 返回 False（Redis 正常），不调用 agent。"""
        bus = _make_bus()
        agent_svc = _make_agent_service(tmp_path)
        adapter = AgentAdapter(bus=bus, agent_service=agent_svc)

        with patch("novare.channels.adapter.redis_service") as mock_redis, \
             patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="u-1"):
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=False)

            msg = _make_msg(metadata={"message_id": "msg-dup"})
            await adapter._handle_one(msg)

        agent_svc.agent.run_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_nx_degraded_continues_processing(self, tmp_path):
        """set_nx 返回 None（降级）时，不丢消息，继续处理。"""
        bus = _make_bus()
        agent_svc = _make_agent_service(tmp_path)
        adapter = AgentAdapter(bus=bus, agent_service=agent_svc)

        with patch("novare.channels.adapter.redis_service") as mock_redis, \
             patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="u-1"):
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=None)

            msg = _make_msg(metadata={"message_id": "msg-degrade"})
            await adapter._handle_one(msg)

        # 降级时仍调用 agent
        agent_svc.agent.run_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_unavailable_continues_processing(self, tmp_path):
        """Redis 不可用时，不做去重，继续处理。"""
        bus = _make_bus()
        agent_svc = _make_agent_service(tmp_path)
        adapter = AgentAdapter(bus=bus, agent_service=agent_svc)

        with patch("novare.channels.adapter.redis_service") as mock_redis, \
             patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="u-1"):
            mock_redis.is_available = False

            msg = _make_msg()
            await adapter._handle_one(msg)

        mock_redis.set_nx.assert_not_called()
        agent_svc.agent.run_turn.assert_awaited_once()

    def test_dedupe_key_from_message_id(self):
        """优先使用 metadata.message_id。"""
        msg = _make_msg(metadata={"message_id": "wx-12345"})
        key = AgentAdapter._build_dedupe_key(msg)
        assert key == "dedupe:channel:test_channel:wx-12345"

    def test_dedupe_key_from_msg_id(self):
        """退化到 metadata.msg_id。"""
        msg = _make_msg(metadata={"msg_id": "tg-999"})
        key = AgentAdapter._build_dedupe_key(msg)
        assert key == "dedupe:channel:test_channel:tg-999"

    def test_dedupe_key_from_id(self):
        """退化到 metadata.id。"""
        msg = _make_msg(metadata={"id": "dc-42"})
        key = AgentAdapter._build_dedupe_key(msg)
        assert key == "dedupe:channel:test_channel:dc-42"

    def test_dedupe_key_fallback_to_content_hash(self):
        """无 message_id 时退化为 content hash。"""
        msg = _make_msg(content="你好世界", metadata={})
        key = AgentAdapter._build_dedupe_key(msg)
        assert key is not None
        assert "dedupe:channel:test_channel:user_1:" in key
        # 同样内容产生同样 key
        msg2 = _make_msg(content="你好世界", metadata={})
        assert AgentAdapter._build_dedupe_key(msg2) == key

    def test_dedupe_key_different_content_different_keys(self):
        """不同内容产生不同 key。"""
        msg1 = _make_msg(content="hello", metadata={})
        msg2 = _make_msg(content="world", metadata={})
        assert AgentAdapter._build_dedupe_key(msg1) != AgentAdapter._build_dedupe_key(msg2)

    @pytest.mark.asyncio
    async def test_dedupe_uses_1h_ttl(self, tmp_path):
        """去重 key 使用 1 小时 TTL。"""
        bus = _make_bus()
        agent_svc = _make_agent_service(tmp_path)
        adapter = AgentAdapter(bus=bus, agent_service=agent_svc)

        with patch("novare.channels.adapter.redis_service") as mock_redis, \
             patch.object(adapter, "_resolve_user", new_callable=AsyncMock, return_value="u-1"):
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)

            msg = _make_msg(metadata={"message_id": "ttl-check"})
            await adapter._handle_one(msg)

        call_args = mock_redis.set_nx.call_args
        assert call_args.kwargs["ttl"] == 3600  # ttl=3600


# ══════════════════════════════════════════════════════════════════════════════
# 5. RedisService — set_json / get_json / expire
# ══════════════════════════════════════════════════════════════════════════════


class TestRedisServiceJson:
    """set_json / get_json / expire 正常和降级行为。"""

    @pytest.fixture
    def svc(self):
        from web.backend.redis_service import RedisService
        return RedisService()

    @pytest.mark.asyncio
    async def test_set_json_and_get_json(self, svc):
        """正常存取 JSON 对象。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            store: dict = {}

            async def fake_set(key, value, **kwargs):
                store[key] = value

            async def fake_get(key):
                return store.get(key)

            mock_client.set = AsyncMock(side_effect=fake_set)
            mock_client.get = AsyncMock(side_effect=fake_get)
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.set_json("task:1", {"status": "running"}) is True
            result = await svc.get_json("task:1")
            assert result == {"status": "running"}

    @pytest.mark.asyncio
    async def test_get_json_returns_none_on_missing(self, svc):
        """key 不存在时 get_json 返回 None。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.get = AsyncMock(return_value=None)
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.get_json("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_json_returns_none_on_decode_error(self, svc):
        """value 不是合法 JSON 时 get_json 返回 None。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.get = AsyncMock(return_value="not-json{{")
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.get_json("bad") is None

    @pytest.mark.asyncio
    async def test_set_json_disabled_returns_false(self, svc):
        """disabled 时 set_json 返回 False。"""
        await svc.initialize(enabled=False, url="")
        assert await svc.set_json("k", {"a": 1}) is False

    @pytest.mark.asyncio
    async def test_get_json_disabled_returns_none(self, svc):
        """disabled 时 get_json 返回 None。"""
        await svc.initialize(enabled=False, url="")
        assert await svc.get_json("k") is None

    @pytest.mark.asyncio
    async def test_expire_ok(self, svc):
        """expire 正常调用。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.expire("k", 60) is True
            mock_client.expire.assert_awaited_once_with("k", 60)

    @pytest.mark.asyncio
    async def test_expire_disabled_returns_false(self, svc):
        """disabled 时 expire 返回 False。"""
        await svc.initialize(enabled=False, url="")
        assert await svc.expire("k", 60) is False


class TestRedisServiceSet:
    """set() 通用写入方法。"""

    @pytest.fixture
    def svc(self):
        from web.backend.redis_service import RedisService
        return RedisService()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, svc):
        """set 带 TTL 正常调用 client.set(ex=ttl)。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.set = AsyncMock()
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.set("k", "v", ttl=120) is True
            mock_client.set.assert_awaited_once_with("k", "v", ex=120)

    @pytest.mark.asyncio
    async def test_set_without_ttl(self, svc):
        """set 不带 TTL 时不传 ex 参数。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.set = AsyncMock()
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.set("k", "v") is True
            mock_client.set.assert_awaited_once_with("k", "v")

    @pytest.mark.asyncio
    async def test_set_disabled_returns_false(self, svc):
        """disabled 时 set 返回 False。"""
        await svc.initialize(enabled=False, url="")
        assert await svc.set("k", "v", ttl=60) is False

    @pytest.mark.asyncio
    async def test_set_exception_returns_false(self, svc):
        """set 异常时返回 False 并标记 unavailable。"""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.set = AsyncMock(side_effect=ConnectionError("broken"))
            mock_from_url.return_value = mock_client
            await svc.initialize(enabled=True, url="redis://localhost:6379/0")

            assert await svc.set("k", "v", ttl=60) is False
            assert svc.is_available is False


# ══════════════════════════════════════════════════════════════════════════════
# 6. AgentService.run_turn — 任务状态写入
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskState:
    """run_turn 的 Redis 任务状态记录行为。"""

    @pytest.fixture(autouse=True)
    def _set_db_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    @pytest.mark.asyncio
    async def test_run_turn_writes_running_state(self, tmp_path):
        """run_turn 开始后写入 running 状态。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-ts", messages=[])
        queue = asyncio.Queue()
        captured_sets: list[tuple] = []

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            async def capture_set_json(key, value, ttl=None):
                captured_sets.append((key, value, ttl))
                return True

            mock_redis.set_json = AsyncMock(side_effect=capture_set_json)
            mock_redis.get = AsyncMock(return_value=None)

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        # 第一次 set_json 应该是 running 状态
        running_calls = [c for c in captured_sets if c[1].get("status") == "running"]
        assert len(running_calls) >= 1
        assert running_calls[0][1]["user_id"] == "u-1"
        assert running_calls[0][1]["session_id"] == "sess-ts"
        # 时间字段不应为空
        assert running_calls[0][1]["started_at"] != ""
        assert running_calls[0][1]["updated_at"] != ""

    @pytest.mark.asyncio
    async def test_run_turn_writes_done_on_success(self, tmp_path):
        """正常完成后写 done 状态。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-done", messages=[])
        queue = asyncio.Queue()
        captured_sets: list[tuple] = []

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            async def capture_set_json(key, value, ttl=None):
                captured_sets.append((key, value, ttl))
                return True

            mock_redis.set_json = AsyncMock(side_effect=capture_set_json)
            mock_redis.get = AsyncMock(return_value=None)

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        done_calls = [c for c in captured_sets if c[1].get("status") == "done"]
        assert len(done_calls) >= 1
        # terminal 状态应保留 started_at
        assert done_calls[0][1]["started_at"] != ""
        assert done_calls[0][1]["updated_at"] != ""

    @pytest.mark.asyncio
    async def test_run_turn_writes_error_on_failure(self, tmp_path):
        """异常后写 error 状态。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(side_effect=RuntimeError("boom"))

        session = MagicMock(session_id="sess-err", messages=[])
        queue = asyncio.Queue()
        captured_sets: list[tuple] = []

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            async def capture_set_json(key, value, ttl=None):
                captured_sets.append((key, value, ttl))
                return True

            mock_redis.set_json = AsyncMock(side_effect=capture_set_json)
            mock_redis.get = AsyncMock(return_value=None)

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        error_calls = [c for c in captured_sets if c[1].get("status") == "error"]
        assert len(error_calls) >= 1
        assert "boom" in error_calls[0][1]["error"]
        # terminal 状态应保留 started_at
        assert error_calls[0][1]["started_at"] != ""
        assert error_calls[0][1]["updated_at"] != ""

    @pytest.mark.asyncio
    async def test_run_turn_skips_state_when_redis_unavailable(self, tmp_path):
        """Redis 不可用时不影响 run_turn。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-noredis", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = False

            result = await svc.run_turn(session, "hello", queue, user_id="u-1")

        svc.agent.run_turn.assert_awaited_once()
        # set_json 不应被调用
        mock_redis.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_terminal_state_not_overwritten_by_async_tool_status(self, tmp_path):
        """异步 tool 状态写入不会覆盖 terminal 状态（gather 保证顺序）。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()

        call_order: list[str] = []

        async def fake_run_turn(session, user_input, on_tool=None, should_cancel=None, **kwargs):
            # 模拟 on_tool("start") — 触发异步状态写入
            if on_tool:
                on_tool("start", "read_file", {}, None, None)
            return "ok"

        svc.agent.run_turn = AsyncMock(side_effect=fake_run_turn)

        session = MagicMock(session_id="sess-order", messages=[])
        queue = asyncio.Queue()

        async def tracking_set_json(key, value, ttl=None):
            call_order.append(value["status"])
            return True

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)
            mock_redis.set_json = AsyncMock(side_effect=tracking_set_json)
            mock_redis.get = AsyncMock(return_value=None)

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        # 最后一个 set_json 调用应是 "done"，不是 "running"
        assert call_order[-1] == "done"


# ══════════════════════════════════════════════════════════════════════════════
# 7. AgentService.run_turn — 协作式取消
# ══════════════════════════════════════════════════════════════════════════════


class TestCooperativeCancellation:
    """协作式取消：should_cancel 闭包 + _cancelled flag。"""

    @pytest.fixture(autouse=True)
    def _set_db_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    @pytest.mark.asyncio
    async def test_cancel_flag_sends_cancelled_and_done(self, tmp_path):
        """cancel key 存在时，queue 收到 cancelled + done。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()

        # AgentLoop.run_turn 模拟：调 should_cancel 后返回
        async def fake_run_turn(*args, should_cancel=None, **kwargs):
            if should_cancel and await should_cancel():
                return "任务已取消。"
            return "ok"

        svc.agent.run_turn = AsyncMock(side_effect=fake_run_turn)

        session = MagicMock(session_id="sess-cancel", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)
            mock_redis.set_json = AsyncMock(return_value=True)
            # cancel key 存在 → should_cancel 返回 True
            mock_redis.get = AsyncMock(return_value="1")

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        events = []
        while not queue.empty():
            events.append(await queue.get())

        cancelled_events = [e for e in events if e.get("type") == "cancelled"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(cancelled_events) == 1
        assert "任务已取消" in cancelled_events[0]["message"]
        assert len(done_events) >= 1

    @pytest.mark.asyncio
    async def test_cancel_writes_cancelled_state(self, tmp_path):
        """取消后写 cancelled 状态到 Redis。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()

        async def fake_run_turn(*args, should_cancel=None, **kwargs):
            if should_cancel and await should_cancel():
                return "任务已取消。"
            return "ok"

        svc.agent.run_turn = AsyncMock(side_effect=fake_run_turn)

        session = MagicMock(session_id="sess-cstate", messages=[])
        queue = asyncio.Queue()
        captured_sets: list[tuple] = []

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set_nx = AsyncMock(return_value=True)
            mock_redis.delete = AsyncMock()
            mock_redis.delete_if_value = AsyncMock(return_value=True)

            async def capture_set_json(key, value, ttl=None):
                captured_sets.append((key, value, ttl))
                return True

            mock_redis.set_json = AsyncMock(side_effect=capture_set_json)
            mock_redis.get = AsyncMock(return_value="1")

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        cancelled_calls = [c for c in captured_sets if c[1].get("status") == "cancelled"]
        assert len(cancelled_calls) >= 1
        # cancelled 状态应保留 started_at
        assert cancelled_calls[0][1]["started_at"] != ""
        assert cancelled_calls[0][1]["updated_at"] != ""

    @pytest.mark.asyncio
    async def test_no_cancel_when_redis_unavailable(self, tmp_path):
        """Redis 不可用时，should_cancel 始终返回 False，不触发取消。"""
        from web.backend.agent_service import AgentService

        svc = AgentService()
        svc.config = MagicMock()
        svc.config.system_prompt = "test"
        svc.config.turn_timeout = 300
        svc._workspace_for = MagicMock(return_value=tmp_path)
        svc.memory_service = None
        svc.agent = AsyncMock()
        svc.agent.run_turn = AsyncMock(return_value="ok")

        session = MagicMock(session_id="sess-noredis", messages=[])
        queue = asyncio.Queue()

        with patch("web.backend.agent_service.redis_service") as mock_redis:
            mock_redis.is_available = False

            await svc.run_turn(session, "hello", queue, user_id="u-1")

        # agent 正常执行，无 cancelled 事件
        svc.agent.run_turn.assert_awaited_once()
        events = []
        while not queue.empty():
            events.append(await queue.get())
        cancelled_events = [e for e in events if e.get("type") == "cancelled"]
        assert len(cancelled_events) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. HTTP cancel 端点
# ══════════════════════════════════════════════════════════════════════════════


class TestCancelEndpoint:
    """POST /api/chat/{session_id}/cancel 端点行为。"""

    @pytest.fixture(autouse=True)
    def _set_db_url(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    def _make_client_with_user(self, user_id: str = "00000000-0000-0000-0000-000000000001"):
        """构造 TestClient，覆盖 get_current_user 依赖避免真实认证。"""
        from fastapi.testclient import TestClient
        from web.backend.app import app
        from web.backend.db.models import User

        fake_user = MagicMock(spec=User)
        fake_user.id = user_id

        async def _override():
            return fake_user

        from web.backend.auth.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = _override
        client = TestClient(app)
        return client

    def test_cancel_redis_available(self):
        """Redis 可用时：调用 set(ttl=...)，返回 ok。"""
        import web.backend.routes.chat as chat_mod
        with patch.object(chat_mod, "redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set = AsyncMock(return_value=True)
            client = self._make_client_with_user()
            try:
                resp = client.post("/api/chat/sess-1/cancel")
            finally:
                from web.backend.app import app
                app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        mock_redis.set.assert_awaited_once()
        _, kwargs = mock_redis.set.call_args
        assert "ttl" in kwargs
        assert "ex" not in kwargs

    def test_cancel_redis_unavailable(self):
        """Redis 不可用时：不调用 set，返回 redis_unavailable。"""
        import web.backend.routes.chat as chat_mod
        with patch.object(chat_mod, "redis_service") as mock_redis:
            mock_redis.is_available = False
            mock_redis.set = AsyncMock(return_value=True)
            client = self._make_client_with_user()
            try:
                resp = client.post("/api/chat/sess-1/cancel")
            finally:
                from web.backend.app import app
                app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["reason"] == "redis_unavailable"
        mock_redis.set.assert_not_awaited()

    def test_cancel_uses_ttl_not_ex(self):
        """确保 set 调用使用 ttl= 而非 ex=（与 RedisService.set 签名一致）。"""
        import web.backend.routes.chat as chat_mod
        with patch.object(chat_mod, "redis_service") as mock_redis:
            mock_redis.is_available = True
            mock_redis.set = AsyncMock(return_value=True)
            client = self._make_client_with_user()
            try:
                client.post("/api/chat/sess-abc/cancel")
            finally:
                from web.backend.app import app
                app.dependency_overrides.clear()

        mock_redis.set.assert_awaited_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == f"cancel:user:00000000-0000-0000-0000-000000000001:session:sess-abc"
        assert args[1] == "1"
        assert "ttl" in kwargs
        assert "ex" not in kwargs
        assert kwargs["ttl"] > 0
