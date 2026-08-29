"""scripts/smoke_postgres_checkpoint.py — 阶段 2 冒烟：真实 Postgres checkpoint

验证链路（AsyncPostgresSaver 后端）：
1. build_checkpointer(DATABASE_URL) 建表并连接
2. 崩溃 turn：工具执行完成 → 模型调用崩溃 → checkpoint 保留在 Postgres
3. resume：从 checkpoint 续跑，已完成的工具不重放（副作用恰好一次）
4. 正常完成 → checkpoint 自动清理

用法：.venv/Scripts/python.exe scripts/smoke_postgres_checkpoint.py
需要 .env 的 DATABASE_URL 指向 PostgreSQL。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from novare.graph.adapters.model import ModelResult, ToolCallSpec  # noqa: E402
from novare.graph.checkpointer import build_checkpointer, dispose_checkpointer  # noqa: E402
from novare.graph.runner import GraphRunner  # noqa: E402
from novare.recovery.state import RecoveryState  # noqa: E402
from novare.session import Session  # noqa: E402
from novare.tools.registry import ToolDef, ToolRegistry  # noqa: E402


class CrashOnNthModel:
    """第 n 次 astream_collect 抛异常（之前的响应正常消费）。"""

    def __init__(self, responses, crash_at: int):
        self._responses = list(responses)
        self._crash_at = crash_at
        self._n = 0

    async def astream_collect(self, messages, tools=None, on_delta=None):
        self._n += 1
        if self._n == self._crash_at:
            raise RuntimeError("boom: simulated transport crash mid-turn")
        return self._responses.pop(0)


class FixedModel:
    def __init__(self, response: ModelResult):
        self._response = response

    async def astream_collect(self, messages, tools=None, on_delta=None):
        return self._response


async def main() -> int:
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgresql://", "postgres://")):
        print("FAIL: .env 未配置 PostgreSQL DATABASE_URL")
        return 1

    checkpointer = await build_checkpointer(database_url)
    try:
        registry = ToolRegistry()
        call_count = {"n": 0}

        async def search_handler(args, **kwargs) -> str:
            call_count["n"] += 1
            return json.dumps({"ok": True, "summary": "3 papers found"}, ensure_ascii=False)

        registry.register_tool(ToolDef(
            name="search_papers", description="smoke", parameters={"type": "object", "properties": {}},
            handler=search_handler,
        ))

        runner = GraphRunner(
            model=CrashOnNthModel([
                ModelResult(content="", tool_calls=[ToolCallSpec(id="call_1", name="search_papers", arguments={"q": "llm"})], stop_reason="tool_calls"),
                ModelResult(content="never reached", tool_calls=[], stop_reason="stop"),
            ], crash_at=2),
            tool_registry=registry,
            system_prompt="smoke",
            max_iterations=5,
            turn_timeout=30,
            auto_compact_threshold=0,
            checkpointer=checkpointer,
        )

        session = Session(session_id="smoke-session", workspace=Path("./workspace"))
        snapshots: list[dict] = []

        # ── 1. 崩溃 turn ──
        try:
            await runner.run_turn(
                session, "search llm papers",
                tool_context={"user_id": "smoke-user"},
                on_recovery_state=lambda snap: snapshots.append(snap),
            )
            print("FAIL: 预期崩溃未发生")
            return 1
        except RuntimeError as exc:
            assert "boom" in str(exc)
        assert call_count["n"] == 1, f"崩溃前工具应执行一次，实际 {call_count['n']}"

        snapshot = snapshots[-1]
        thread_id = snapshot["thread_id"]
        assert thread_id.startswith("smoke-user:smoke-session:"), thread_id

        # checkpoint 应存在于 Postgres
        state = runner.get_thread_state(thread_id)
        assert state.values, "崩溃后 checkpoint 应保留在 Postgres"
        print(f"✓ 崩溃后 checkpoint 已持久化（thread={thread_id[:40]}...）")

        # ── 2. resume ──
        runner.model = FixedModel(ModelResult(content="Found 3 papers.", tool_calls=[], stop_reason="stop"))
        answer = await runner.run_turn(
            session, "search llm papers",
            tool_context={"user_id": "smoke-user"},
            resume=True,
            thread_id=thread_id,
            recovery_state=RecoveryState.from_dict(snapshot),
        )
        assert answer == "Found 3 papers.", answer
        assert call_count["n"] == 1, f"resume 后不得重放工具，实际执行 {call_count['n']} 次"
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        print("✓ resume 从 checkpoint 续跑完成，工具副作用恰好一次")

        # ── 3. completed 清理 ──
        state = runner.get_thread_state(thread_id)
        assert not state.values, "completed 后 checkpoint 应被清理"
        print("✓ completed 的 checkpoint 已从 Postgres 清理")

        print("\n冒烟通过：AsyncPostgresSaver 全链路（建表/写入/恢复/清理）OK")
        return 0
    finally:
        await dispose_checkpointer(checkpointer)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
