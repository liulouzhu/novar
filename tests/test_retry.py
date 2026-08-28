"""tests/test_retry.py — PR 1：LLM / 工具重试集成测试

覆盖：
- LLM 前两次 503、第三次成功，只产生一条 assistant 消息
- LLM 400/401 只调用一次（确定性错误不重试）
- LLM 输出部分文本后断流，不透明重试
- asyncio.CancelledError 绝不重试
- 只读工具瞬时失败后成功
- 工具参数错误不重试
- 非幂等工具遇到瞬时错误也不重试
- 工具重试耗尽后只有一个 tool result，attempts/error_code 正确
- Retry-After 被采用并受上限限制
- 每轮共享 retry budget 生效
- LLM 重试耗尽抛 RetryExhaustedError（带原始异常链）
"""

import asyncio
import json

import httpx
import pytest
from httpx import HTTPStatusError, Request, Response
from unittest.mock import AsyncMock

from novare.agent_loop import AgentLoop
from novare.llm_client import LLMResponse, ToolCall
from novare.recovery import RetryExhaustedError, RetryPolicy
from novare.session import Session
from novare.tools.registry import ToolDef, ToolRegistry


def _status_error(status_code: int, retry_after: str | None = None) -> HTTPStatusError:
    req = Request("POST", "http://example.com/chat/completions")
    headers = {"retry-after": retry_after} if retry_after else None
    resp = Response(status_code, request=req, headers=headers)
    return HTTPStatusError(f"HTTP {status_code}", request=req, response=resp)


def _make_loop(
    llm,
    *,
    registry: ToolRegistry | None = None,
    retry_sleep=None,
    max_retries_per_turn: int = 6,
    retry_after_max_delay: float = 30.0,
    retry_base_delay: float = 0.5,
):
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    loop = AgentLoop(
        llm_client=llm,
        tool_registry=registry or ToolRegistry(),
        system_prompt="You are a test assistant.",
        retry_sleep=retry_sleep or _fake_sleep,
        retry_after_max_delay=retry_after_max_delay,
        retry_base_delay=retry_base_delay,
        max_retries_per_turn=max_retries_per_turn,
    )
    return loop, sleeps


class TestLLMRetry:
    @pytest.mark.asyncio
    async def test_503_twice_then_success_single_assistant_message(self):
        """前两次 503、第三次成功：只产生一条 assistant 消息，失败 attempt 不写 session。"""
        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            _status_error(503),
            _status_error(503),
            LLMResponse(content="Hello!", tool_calls=[], stop_reason="stop", usage={}),
        ])
        loop, sleeps = _make_loop(llm)
        session = Session()
        captured = []

        result = await loop.run_turn(session, "Hi", on_message=lambda m: captured.append(m))

        assert result == "Hello!"
        assert llm.collect_stream.await_count == 3
        assert len(sleeps) == 2
        # 失败 attempt 不写 session / 不调用 on_message：只有 user + assistant 两条
        assert [m["role"] for m in session.messages] == ["user", "assistant"]
        assert [m["role"] for m in captured] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_deterministic_errors_not_retried(self):
        """400 / 401 确定性错误只调用一次。"""
        for code in (400, 401):
            llm = AsyncMock()
            llm.collect_stream = AsyncMock(side_effect=[_status_error(code)])
            loop, sleeps = _make_loop(llm)
            session = Session()
            with pytest.raises(HTTPStatusError):
                await loop.run_turn(session, "Hi")
            assert llm.collect_stream.await_count == 1
            assert sleeps == []

    @pytest.mark.asyncio
    async def test_no_retry_after_partial_stream_output(self):
        """输出部分文本后断流：不得透明重试（避免前端重复文本）。"""
        async def flaky_stream(messages, tools=None, max_tokens=4096, on_text=None):
            on_text("partial output")
            raise httpx.ReadTimeout("stream broke")

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=flaky_stream)
        loop, sleeps = _make_loop(llm)
        session = Session()
        texts = []

        with pytest.raises(httpx.ReadTimeout):
            await loop.run_turn(session, "Hi", on_text=texts.append)

        assert texts == ["partial output"]
        assert llm.collect_stream.await_count == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_cancelled_error_not_retried(self):
        """asyncio.CancelledError 立即传播，绝不重试。"""
        async def cancel_stream(messages, **kwargs):
            raise asyncio.CancelledError()

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=cancel_stream)
        loop, sleeps = _make_loop(llm)
        session = Session()

        with pytest.raises(asyncio.CancelledError):
            await loop.run_turn(session, "Hi")

        assert llm.collect_stream.await_count == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_retry_after_used_and_capped(self):
        """Retry-After 被采用，且受 retry_after_max_delay 上限约束。"""
        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            _status_error(503, retry_after="999"),
            LLMResponse(content="ok", tool_calls=[], stop_reason="stop"),
        ])
        loop, sleeps = _make_loop(llm, retry_after_max_delay=5.0)

        session = Session()
        result = await loop.run_turn(session, "Hi")

        assert result == "ok"
        assert sleeps == [5.0]  # min(999, 5)

    @pytest.mark.asyncio
    async def test_llm_retry_exhausted_raises_chain(self):
        """重试耗尽后抛 RetryExhaustedError，携带诊断字段且不保留原始异常对象。"""
        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            _status_error(503), _status_error(503), _status_error(503),
        ])
        loop, sleeps = _make_loop(llm)

        session = Session()
        with pytest.raises(RetryExhaustedError) as exc_info:
            await loop.run_turn(session, "Hi")

        err = exc_info.value
        assert err.attempts == 3
        # 不再通过 __cause__ 保留原始异常（避免 traceback 泄密）
        assert err.__cause__ is None
        # 非敏感诊断字段可用于判断原始错误类别
        assert err.cause_type == "HTTPStatusError"
        assert err.status_code == 503
        assert err.error_code == "SERVICE_UNAVAILABLE"
        assert len(sleeps) == 2

    @pytest.mark.asyncio
    async def test_run_reviewer_uses_reviewer_llm_not_main(self):
        """run_reviewer 必须调用 reviewer_llm（独立 client），主模型不被调用。"""
        main_llm = AsyncMock()
        main_llm.collect_stream = AsyncMock()
        reviewer_llm = AsyncMock()
        reviewer_llm.collect_stream = AsyncMock(side_effect=[
            _status_error(503),
            LLMResponse(content='{"reviews": []}', tool_calls=[], stop_reason="stop"),
        ])
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        loop = AgentLoop(
            llm_client=main_llm,
            reviewer_llm=reviewer_llm,
            tool_registry=ToolRegistry(),
            system_prompt="You are a test assistant.",
            retry_sleep=fake_sleep,
        )

        result = await loop.run_reviewer("evaluate this idea")

        # reviewer 上完成 503 → 重试 → 成功
        assert reviewer_llm.collect_stream.await_count == 2
        assert len(sleeps) == 1
        assert "reviews" in result
        # 主模型完全未被调用
        assert main_llm.collect_stream.await_count == 0


    @pytest.mark.asyncio
    async def test_retry_exhausted_message_is_sanitized(self):
        """RetryExhaustedError 消息不泄漏 API key 等敏感信息。"""

        class SensitiveError(Exception):
            status_code = 503

            def __str__(self):
                return "upstream timeout with api_key=sk-abcdefgh12345678"

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            SensitiveError("x"), SensitiveError("x"), SensitiveError("x"),
        ])
        loop, _ = _make_loop(llm)
        session = Session()

        with pytest.raises(RetryExhaustedError) as exc_info:
            await loop.run_turn(session, "Hi")

        message = str(exc_info.value)
        assert "sk-abcdefgh12345678" not in message
        assert "abcdefgh12345678" not in message
        assert "***" in message  # 已被脱敏标记替换

    @pytest.mark.asyncio
    async def test_retry_exhausted_never_leaks_secret(self, caplog):
        """RetryExhaustedError 的 str/repr/traceback/日志均不含 secret，
        __cause__ 不指向原始敏感异常，且可通过 cause_type/status_code 判断类别。"""
        import logging
        import traceback

        class SensitiveError(Exception):
            status_code = 503

            def __str__(self):
                return "Authorization: Bearer abcdefgh api_key=sk-abcdefgh12345678"

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            SensitiveError("x"), SensitiveError("x"), SensitiveError("x"),
        ])
        loop, _ = _make_loop(llm)
        session = Session()

        with pytest.raises(RetryExhaustedError) as exc_info:
            await loop.run_turn(session, "Hi")
        err = exc_info.value

        # str / repr
        assert "abcdefgh12345678" not in str(err)
        assert "abcdefgh" not in str(err)
        assert "abcdefgh12345678" not in repr(err)
        assert "abcdefgh" not in repr(err)

        # __cause__ 不得指向原始敏感异常
        assert err.__cause__ is None

        # traceback.format_exception 不含 secret
        tb = traceback.format_exception(type(err), err, err.__traceback__)
        assert "abcdefgh12345678" not in "".join(tb)
        assert "abcdefgh" not in "".join(tb)

        # 非敏感诊断字段仍可判断原始错误类别
        assert err.cause_type == "SensitiveError"
        assert err.status_code == 503
        assert err.error_code == "SERVICE_UNAVAILABLE"

        # logger.exception 输出不含 secret（第二次运行前重置 side_effect）
        llm.collect_stream.side_effect = [
            SensitiveError("x"), SensitiveError("x"), SensitiveError("x"),
        ]
        with caplog.at_level(logging.ERROR):
            try:
                await loop.run_turn(session, "Hi")
            except RetryExhaustedError:
                logging.getLogger("test.retry.leak").exception("run_turn failed")
        assert "abcdefgh12345678" not in caplog.text
        assert "abcdefgh" not in caplog.text

    @pytest.mark.asyncio
    async def test_sanitize_error_forms(self):
        """sanitize_error 完整脱敏 Authorization/Bearer/api_key/裸 token 各种形式。"""
        from novare.recovery.classifier import sanitize_error

        cases = [
            "Authorization: Bearer abcdefgh",
            "Authorization=Basic abcdefgh",
            "Bearer abcdefgh",
            "api_key=sk-abcdefgh12345678",
            "API-KEY: sk-abcdefgh12345678",
            "token sk-abcdefgh12345678 in text",
            "https://example.com/export?key=sk-abcdefgh12345678",
        ]
        for raw in cases:
            out = sanitize_error(raw)
            assert "abcdefgh12345678" not in out, (raw, out)
            assert "abcdefgh" not in out, (raw, out)
            assert "***" in out, (raw, out)


class TestToolRetry:
    def _tool_loop(self, tool_def, llm_responses, **kwargs):
        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=llm_responses)
        registry = ToolRegistry()
        registry.register_tool(tool_def)
        loop, sleeps = _make_loop(llm, registry=registry, **kwargs)
        return loop, sleeps, registry

    @pytest.mark.asyncio
    async def test_read_tool_transient_failure_then_success(self):
        """只读工具瞬时失败后成功：自动重试，只写一个 tool result。"""
        calls = []

        async def flaky_handler(args, workspace=None):
            calls.append(1)
            if len(calls) == 1:
                return "Error executing reader: upstream timeout"
            return "ok result"

        loop, sleeps, _ = self._tool_loop(
            ToolDef(
                name="reader", description="r", parameters={},
                handler=flaky_handler, idempotency="read",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go")

        assert result == "done"
        assert len(calls) == 2
        assert len(sleeps) == 1
        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio
    async def test_semantic_error_not_retried(self):
        """工具参数错误（SEMANTIC）不重试。"""
        calls = []

        async def bad_args(args, workspace=None):
            calls.append(1)
            return "Error: Invalid parameter 'query'"

        loop, sleeps, _ = self._tool_loop(
            ToolDef(
                name="reader", description="r", parameters={},
                handler=bad_args, idempotency="read",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []
        tool_content = session.messages[2]["content"]
        parsed = json.loads(tool_content)
        assert parsed["error_code"] == "INVALID_ARGUMENT"
        assert parsed["attempts"] == 1
        assert parsed["outcome"] == "not_applied"

    @pytest.mark.asyncio
    async def test_non_idempotent_tool_not_retried_even_on_transient(self):
        """非幂等工具遇到瞬时错误也不重试（不假设“超时等于未执行”）。"""
        calls = []

        async def writer(args, workspace=None):
            calls.append(1)
            return "Error executing writer: connection reset"

        loop, sleeps, _ = self._tool_loop(
            ToolDef(name="writer", description="w", parameters={}, handler=writer),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="writer", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []
        parsed = json.loads(session.messages[2]["content"])
        assert parsed["error_code"] == "CONNECTION_RESET"
        assert parsed["retryable"] is True
        assert parsed["outcome"] == "not_applied"
        assert parsed["attempts"] == 1

    @pytest.mark.asyncio
    async def test_tool_retry_exhausted_single_result(self):
        """工具重试耗尽：只有一个 tool result，attempts/error_code 正确。"""
        calls = []

        async def always_fail(args, workspace=None):
            calls.append(1)
            return "Error executing paper_search: upstream timeout"

        events = []

        def on_tool(event, name, args, result, elapsed):
            events.append((event, name, result))

        loop, sleeps, _ = self._tool_loop(
            ToolDef(
                name="paper_search", description="p", parameters={},
                handler=always_fail, idempotency="read",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="paper_search", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go", on_tool=on_tool)

        assert result == "done"
        assert len(calls) == 3
        assert len(sleeps) == 2
        tool_msgs = [m for m in session.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1

        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed["ok"] is False
        assert parsed["error_code"] == "UPSTREAM_TIMEOUT"
        assert parsed["retryable"] is True
        assert parsed["outcome"] == "retry_exhausted"
        assert parsed["attempts"] == 3
        assert "upstream timeout" in parsed["error"]

        # on_tool 事件：start → retry → retry → error（保持兼容签名）
        assert [e[0] for e in events] == ["start", "retry", "retry", "error"]
        retry_info = json.loads(events[1][2])
        assert retry_info["attempt"] == 1
        assert retry_info["max_attempts"] == 3
        assert retry_info["error_code"] == "UPSTREAM_TIMEOUT"

    @pytest.mark.asyncio
    async def test_shared_retry_budget_per_turn(self):
        """每轮共享 retry budget：第一个工具消耗预算后，第二个工具无法再重试。"""
        a_calls = []
        b_calls = []

        async def tool_a(args, workspace=None):
            a_calls.append(1)
            if len(a_calls) == 1:
                return "Error executing reader: upstream timeout"
            return "a ok"

        async def tool_b(args, workspace=None):
            b_calls.append(1)
            return "Error executing reader: upstream timeout"

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_a", name="tool_a", arguments={}),
            ], stop_reason="tool_calls", usage={}),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc_b", name="tool_b", arguments={}),
            ], stop_reason="tool_calls", usage={}),
            LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
        ])
        registry = ToolRegistry()
        registry.register_tool(ToolDef(
            name="tool_a", description="a", parameters={}, handler=tool_a,
            idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
        ))
        registry.register_tool(ToolDef(
            name="tool_b", description="b", parameters={}, handler=tool_b,
            idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
        ))
        loop, sleeps = _make_loop(llm, registry=registry, max_retries_per_turn=1)

        session = Session()
        result = await loop.run_turn(session, "go")

        assert result == "done"
        # tool_a 用掉唯一一次重试预算后成功；tool_b 预算耗尽 → 不重试
        assert len(a_calls) == 2
        assert len(b_calls) == 1
        assert len(sleeps) == 1


class TestNonIdempotentEnforcement:
    """问题二：AgentLoop 强制非幂等保护（不依赖工具名作为安全判断）。"""

    @pytest.mark.asyncio
    async def test_non_idempotent_with_max3_not_retried(self):
        """non_idempotent + RetryPolicy(max_attempts=3) + transient failure → 只执行一次。"""
        calls = []

        async def writer(args, workspace=None):
            calls.append(1)
            return "Error executing writer: connection reset"

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="writer", description="w", parameters={}, handler=writer,
                idempotency="non_idempotent",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="writer", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []
        parsed = json.loads(session.messages[2]["content"])
        assert parsed["error_code"] == "CONNECTION_RESET"
        assert parsed["attempts"] == 1

    @pytest.mark.asyncio
    async def test_idempotent_write_allows_retry(self):
        """idempotent_write 可按声明策略重试。"""
        calls = []

        async def writer(args, workspace=None):
            calls.append(1)
            if len(calls) == 1:
                return "Error executing writer: connection reset"
            return "OK: written"

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="writer", description="w", parameters={}, handler=writer,
                idempotency="idempotent_write",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="writer", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go")

        assert result == "done"
        assert len(calls) == 2
        assert len(sleeps) == 1

    @pytest.mark.asyncio
    async def test_fake_executor_without_idempotency_for_not_retried(self):
        """fake executor 只有 retry_policy_for()、没有 idempotency_for() → 保守只执行一次。"""
        class RetryOnlyExecutor:
            def __init__(self):
                self.calls = []

            def to_openai_tools(self):
                return []

            def retry_policy_for(self, name):
                return RetryPolicy(max_attempts=3)

            async def execute(self, name, arguments, tool_context=None):
                self.calls.append(1)
                return "Error executing x: connection reset"

        llm = AsyncMock()
        llm.collect_stream = AsyncMock(side_effect=[
            LLMResponse(content="", tool_calls=[
                ToolCall(id="tc1", name="x", arguments={}),
            ], stop_reason="tool_calls", usage={}),
            LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
        ])
        executor = RetryOnlyExecutor()
        loop, sleeps = _make_loop(llm, registry=executor)

        session = Session()
        await loop.run_turn(session, "go")

        assert executor.calls == [1]
        assert sleeps == []


class TestStructuredVeto:
    """问题三：结构化 retryable/outcome 拥有否决权。"""

    def _structured_failure(self, **fields):
        base = {
            "ok": False,
            "error": "Error executing reader: upstream timeout",
            "error_code": "TIMEOUT",
            "retryable": True,
            "outcome": "not_applied",
        }
        base.update(fields)
        return json.dumps(base, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_retryable_false_blocks_retry(self):
        """TIMEOUT + retryable:false + outcome:not_applied → 不重试（error_code 不得升级）。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            return self._structured_failure(retryable=False)

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_outcome_unknown_blocks_retry(self):
        """TIMEOUT + retryable:true + outcome:unknown → 不重试。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            return self._structured_failure(outcome="unknown")

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_outcome_retry_exhausted_no_nested_retry(self):
        """TIMEOUT + retryable:true + outcome:retry_exhausted → 不发生嵌套重试。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            return self._structured_failure(outcome="retry_exhausted")

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_retryable_true_not_applied_read_allows_retry(self):
        """TIMEOUT + retryable:true + outcome:not_applied + read → 允许重试。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            if len(calls) == 1:
                return self._structured_failure(retryable=True, outcome="not_applied")
            return "ok result"

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go")

        assert result == "done"
        assert len(calls) == 2
        assert len(sleeps) == 1

    @pytest.mark.asyncio
    async def test_same_result_non_idempotent_not_retried(self):
        """同一结构化结果用于 non_idempotent 工具时仍不得重试。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            return self._structured_failure(retryable=True, outcome="not_applied")

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="non_idempotent",
                retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 1
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_string_false_is_not_trusted(self):
        """字符串 "false" 不被当作布尔 False（bool("false") 是 True）。"""
        calls = []

        async def handler(args, workspace=None):
            calls.append(1)
            if len(calls) == 1:
                return self._structured_failure(retryable="false")  # 字符串，非法
            return "ok result"

        loop, sleeps, _ = _tool_loop_with(
            ToolDef(
                name="reader", description="r", parameters={}, handler=handler,
                idempotency="read", retry_policy=RetryPolicy(max_attempts=3),
            ),
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="reader", arguments={}),
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        # 字符串 "false" 不信任 → 按缺失处理（transient → read 允许重试）
        assert len(calls) == 2


def _tool_loop_with(tool_def, llm_responses, **kwargs):
    """独立于 TestToolRetry 的工具循环构造（供 TestStructuredVeto / TestNonIdempotentEnforcement 使用）。"""
    from unittest.mock import AsyncMock as _AM

    llm = _AM()
    llm.collect_stream = _AM(side_effect=llm_responses)
    registry = ToolRegistry()
    registry.register_tool(tool_def)
    loop, sleeps = _make_loop(llm, registry=registry, **kwargs)
    return loop, sleeps, registry
