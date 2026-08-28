"""tests/test_health.py — /api/health 端点测试

覆盖：
1. Redis disabled 时返回 status=disabled
2. Redis 正常时返回 status=ok
3. Redis unavailable 时返回 status=unavailable
4. DB 查询异常时接口不 500
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _set_db_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


def _make_test_client():
    """构造 FastAPI TestClient（lazy import 避免循环）。"""
    from fastapi.testclient import TestClient
    from web.backend.app import app
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_redis_disabled(self):
        """Redis disabled → enabled=false, status=disabled。"""
        with patch("web.backend.app.redis_service") as mock_redis:
            mock_redis._enabled = False
            mock_redis.is_available = False
            client = _make_test_client()
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis"]["enabled"] is False
        assert data["redis"]["available"] is False
        assert data["redis"]["status"] == "disabled"

    def test_health_redis_ok(self):
        """Redis ping 成功 → status=ok。"""
        with patch("web.backend.app.redis_service") as mock_redis:
            mock_redis._enabled = True
            mock_redis.is_available = True
            mock_redis.ping = AsyncMock(return_value=True)
            client = _make_test_client()
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis"]["enabled"] is True
        assert data["redis"]["available"] is True
        assert data["redis"]["status"] == "ok"

    def test_health_redis_unavailable(self):
        """Redis ping 失败 → status=unavailable。"""
        with patch("web.backend.app.redis_service") as mock_redis:
            mock_redis._enabled = True
            mock_redis.is_available = False
            mock_redis.ping = AsyncMock(return_value=False)
            client = _make_test_client()
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis"]["status"] == "unavailable"

    def test_health_db_error_no_500(self):
        """DB 查询异常时接口仍返回 200。"""
        with patch("web.backend.app.redis_service") as mock_redis:
            mock_redis._enabled = False
            mock_redis.is_available = False
            # Mock get_session_factory to return a factory that yields a failing session
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(side_effect=Exception("connection refused"))

            class MockFactory:
                def __call__(self):
                    return self
                async def __aenter__(self):
                    return mock_session
                async def __aexit__(self, *args):
                    return False

            import web.backend.app as app_mod
            original_factory = app_mod.get_session_factory
            app_mod.get_session_factory = MockFactory
            try:
                client = _make_test_client()
                resp = client.get("/api/health")
            finally:
                app_mod.get_session_factory = original_factory
        assert resp.status_code == 200
        data = resp.json()
        assert data["database"]["status"] == "error"
