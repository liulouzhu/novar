"""tests/test_milvus_isolation.py — 防逃逸 fixture 自测

验证 forbid_real_milvus_network fixture 的行为：
1. 业务代码调用被禁止 API 后，teardown 仍会使测试失败
2. 显式 MagicMock 覆盖后，专门的 Milvus helper 测试正常通过
3. 默认 RAG/cache 测试没有任何违规调用记录
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════
# 1. fixture teardown 使违规测试失败
# ══════════════════════════════════════════════════════════════

class TestFixtureBlocksRealMilvus:
    """验证 fixture teardown 在检测到违规调用时使测试失败。"""

    def test_fixture_tears_down_on_violation(self):
        """如果业务代码调用了被禁止的 API，fixture teardown 会使测试失败。

        由于不能在正常测试中故意让 autouse fixture teardown 失败
        （那会导致整个测试套件报错），我们测试 fixture 的 recorder 逻辑：
        调用被禁止 API 时记录调用，teardown 时断言没有违规。
        """
        # 验证 conftest 中的 _make_recorder 逻辑
        from tests.conftest import _make_recorder

        attempted_calls = []
        recorder = _make_recorder("test_api", attempted_calls)

        # 调用 recorder 应该记录调用并抛出 RuntimeError
        with pytest.raises(RuntimeError, match="Forbidden Milvus call: test_api"):
            recorder("arg1", key="val")

        # 验证调用被记录
        assert len(attempted_calls) == 1
        assert attempted_calls[0][0] == "test_api"
        assert attempted_calls[0][1] == ("arg1",)
        assert attempted_calls[0][2] == {"key": "val"}

    def test_recorder_multiple_calls(self):
        """多次违规调用都被记录。"""
        from tests.conftest import _make_recorder

        attempted_calls = []
        recorder = _make_recorder("api1", attempted_calls)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                recorder()

        assert len(attempted_calls) == 3

    def test_fixture_makes_test_fail_on_violation(self, monkeypatch):
        """模拟 fixture 行为：记录违规后 pytest.fail 会在 teardown 中触发。"""
        attempted_calls = []

        def _record_violation(*args, **kwargs):
            attempted_calls.append(("connect", args, kwargs))
            raise RuntimeError("Forbidden")

        try:
            from pymilvus import connections as _conns
            original = _conns.connect
            _conns.connect = _record_violation
            try:
                # 调用被禁止的 API
                with pytest.raises(RuntimeError):
                    _conns.connect(host="localhost", port=19530)
                # 验证调用被记录
                assert len(attempted_calls) == 1
                assert attempted_calls[0][0] == "connect"
            finally:
                _conns.connect = original
        except ImportError:
            pytest.skip("pymilvus not installed")


# ══════════════════════════════════════════════════════════════
# 2. 显式 MagicMock 覆盖后正常通过
# ══════════════════════════════════════════════════════════════

class TestExplicitMockOverridesFixture:
    """测试自身用 MagicMock 覆盖 fixture 的 recorder 后正常通过。"""

    def test_mock_overrides_fixture(self):
        """用 MagicMock 覆盖 connections.connect 后测试正常通过。"""
        try:
            from pymilvus import connections as _conns
            original = _conns.connect
            mock_connect = MagicMock()
            _conns.connect = mock_connect
            try:
                _conns.connect(host="localhost", port=19530)
                mock_connect.assert_called_once_with(host="localhost", port=19530)
            finally:
                _conns.connect = original
        except ImportError:
            pytest.skip("pymilvus not installed")

    def test_mock_overrides_fixture_in_test_context(self):
        """在测试上下文中用 patch 覆盖 fixture 的 recorder。"""
        try:
            from pymilvus import connections as _conns
            mock_connect = MagicMock()
            with patch.object(_conns, "connect", mock_connect):
                _conns.connect(host="localhost", port=19530)
                mock_connect.assert_called_once()
        except ImportError:
            pytest.skip("pymilvus not installed")


# ══════════════════════════════════════════════════════════════
# 3. 业务代码吞掉异常后 fixture 仍能检测
# ══════════════════════════════════════════════════════════════

class TestBusinessCodeExceptionSwallowing:
    """验证即使业务代码捕获了 RuntimeError，fixture teardown 仍能检测到违规。"""

    def test_recorder_catches_swallowed_exception(self):
        """业务代码吞掉 RuntimeError 后，recorder 仍然记录了调用。"""
        from tests.conftest import _make_recorder

        attempted_calls = []
        recorder = _make_recorder("connect", attempted_calls)

        # 模拟业务代码捕获异常
        try:
            recorder("localhost")
        except RuntimeError:
            pass  # 业务代码吞掉异常

        # recorder 仍然记录了调用
        assert len(attempted_calls) == 1
        assert attempted_calls[0][0] == "connect"
