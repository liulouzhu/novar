"""novare/cli.py — 交互式 REPL 入口"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from novare.config import NovareConfig, get_user_workspace
from novare.llm_client import LLMClient
from novare.session import Session
from novare.tools.registry import ToolRegistry, ToolDef
from novare.mcp_client import McpClient
from novare.agent_loop import AgentLoop
from novare.hallucination_verifier import HallucinationVerifier
from novare.skill import Skill, discover_skills
from novare.subagents.registry import SubagentRegistry
from novare.subagents.tools import register_subagent_tools


# ── 工具状态显示 ────────────────────────────────────────────────────────────

def _format_tool_args(args: dict) -> str:
    """精简展示工具参数：取前 2 个 key，string 截断到 40 字符"""
    items = list(args.items())[:2]
    parts = []
    for k, v in items:
        val = str(v)
        if len(val) > 40:
            val = val[:37] + "..."
        parts.append(f"{k}={val!r}")
    return ", ".join(parts)


def _summarize_result(name: str, result: str | None) -> str:
    """按工具类型生成结果摘要"""
    if not result:
        return "完成"
    if name == "paper_search":
        # 尝试提取论文数量
        import re
        m = re.search(r"找到\s*(\d+)\s*篇|(\d+)\s*results?|共\s*(\d+)", result, re.IGNORECASE)
        if m:
            n = next(g for g in m.groups() if g)
            return f"{n} 条结果"
        return "搜索完成"
    if name == "paper_parse":
        return "解析完成"
    if name == "rag_query":
        import re
        m = re.search(r"(\d+)\s*(?:条|个|results?|matches)", result, re.IGNORECASE)
        if m:
            return f"{m.group(1)} 条匹配"
        return "检索完成"
    if name == "knowledge_graph":
        return "图谱操作完成"
    if name == "code_execute":
        return "执行完成"
    return "完成"


def _make_tool_handler():
    """返回工具状态回调函数"""
    def handler(event: str, name: str, args: dict, result: str | None, elapsed: float | None):
        if event == "start":
            arg_str = _format_tool_args(args)
            print(f"  ⚡ {name}({arg_str})")
        elif event == "end":
            summary = _summarize_result(name, result)
            time_str = f"{elapsed:.1f}s" if elapsed else ""
            print(f"  ✅ {name} · {summary} · {time_str}")
        elif event == "error":
            time_str = f"{elapsed:.1f}s" if elapsed else ""
            err_msg = (result[:60] + "...") if result and len(result) > 60 else (result or "unknown")
            print(f"  ❌ {name} · {err_msg} · {time_str}")
    return handler


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("novare")

    # 加载配置
    config = NovareConfig.load()
    if not config.api_key:
        print("Error: NOVARE_API_KEY not set.", file=sys.stderr)
        print("Set it via: export NOVARE_API_KEY=your-key", file=sys.stderr)
        sys.exit(1)

    logger.info("Novare starting — model=%s, workspace=%s", config.model, config.workspace)

    # 初始化 LLM 客户端
    llm_client = LLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        proxy=config.proxy,
    )

    # 初始化工具注册表
    tool_registry = ToolRegistry(workspace=config.workspace)

    # 连接 MCP 服务器（研究工具）
    mcp_clients: list[McpClient] = []
    # CLI 模式下从环境变量读取 user_id，注入到所有 MCP 工具调用
    cli_user_id = os.environ.get("NOVARE_USER_ID")
    if cli_user_id:
        logger.info("CLI user_id: %s (from NOVARE_USER_ID)", cli_user_id[:8] + "...")
    else:
        logger.info(
            "NOVARE_USER_ID not set — RAG research tools will be unavailable. "
            "Set NOVARE_USER_ID to your user UUID to enable RAG retrieval."
        )

    for name, srv_config in config.mcp_servers.items():
        logger.info("Connecting to MCP server: %s", name)
        client = McpClient(
            command=srv_config.command,
            args=srv_config.args,
            env=srv_config.env,
        )
        try:
            await client.connect()
            tools = await client.list_tools()
            for tool in tools:
                tool_name = tool["name"]

                async def make_handler(c: McpClient, tn: str, uid: str | None):
                    async def handler(args, workspace=None, **_context):
                        # AgentLoop 会把幂等重试相关的内部上下文（例如
                        # ``_idempotency_key``、``_action_fingerprint``）作为
                        # 关键字参数透传给 MCP handler。它们只供 Agent 内部使用，
                        # 不应混入 MCP 的业务参数；这里吸收并忽略这些上下文。
                        # 注入 _user_id（来自可信 tool context，模型无法覆盖）
                        payload = dict(args)
                        if uid:
                            payload["_user_id"] = uid
                        return await c.call_tool(tn, payload)
                    return handler

                handler = await make_handler(client, tool_name, cli_user_id)
                tool_registry.register_tool(ToolDef(
                    name=tool_name,
                    description=tool.get("description", ""),
                    parameters=tool.get("inputSchema", {"type": "object", "properties": {}}),
                    handler=handler,
                    source=f"mcp:{name}",
                ))
            logger.info("MCP server '%s': %d tools registered", name, len(tools))
            mcp_clients.append(client)
        except Exception as e:
            logger.error("Failed to connect to MCP server '%s': %s", name, e)

    hallucination_verifier = None
    if config.hallucination_verifier_enabled:
        hallucination_verifier = HallucinationVerifier(
            llm_client=llm_client,
            tool_executor=tool_registry,
            enabled=True,
            max_claims=config.hallucination_verifier_max_claims,
            top_k=config.hallucination_verifier_top_k,
            max_concurrency=config.hallucination_verifier_concurrency,
            timeout=config.hallucination_verifier_timeout,
        )

    # 创建 AgentLoop
    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=tool_registry,
        system_prompt=config.system_prompt,
        max_iterations=config.max_iterations,
        auto_compact_threshold=config.auto_compact_threshold,
        context_max_turns=config.context_max_turns,
        context_token_budget=config.context_token_budget,
        context_summary_max_tokens=config.context_summary_max_tokens,
        context_tool_result_max_tokens=config.context_tool_result_max_tokens,
        context_llm_timeout=config.context_llm_timeout,
        context_llm_enabled=config.context_llm_enabled,
        hallucination_verifier=hallucination_verifier,
        turn_timeout=config.turn_timeout,
        llm_retry_attempts=config.llm_retry_attempts,
        retry_base_delay=config.retry_base_delay,
        retry_max_delay=config.retry_max_delay,
        max_retries_per_turn=config.max_retries_per_turn,
        retry_after_max_delay=config.retry_after_max_delay,
        reflexion_enabled=config.reflexion_enabled,
        max_reflections_per_turn=config.max_reflections_per_turn,
        reflexion_no_progress_threshold=config.reflexion_no_progress_threshold,
        reflexion_repeated_failure_threshold=config.reflexion_repeated_failure_threshold,
        reflexion_timeout=config.reflexion_timeout,
        reflexion_max_tokens=config.reflexion_max_tokens,
        reflexion_max_recent_events=config.reflexion_max_recent_events,
    )

    # 初始化子智能体系统
    subagent_registry = SubagentRegistry()
    register_subagent_tools(
        tool_registry=tool_registry,
        subagent_registry=subagent_registry,
        llm_client=llm_client,
        system_prompt=config.system_prompt,
        workspace=config.workspace,
        default_max_iterations=config.subagent_max_iterations,
        turn_timeout=config.subagent_turn_timeout,
    )

    # 发现 Skills：系统公共 + 用户私有
    skill_dirs = list(config.skill_dirs)
    user_id = os.environ.get("NOVARE_USER_ID")
    if user_id:
        user_skill_dir = Path(get_user_workspace(user_id)) / ".novare" / "skills"
        skill_dirs.insert(0, user_skill_dir)  # 用户私有优先级最高
    skills = discover_skills(skill_dirs)
    skill_map: dict[str, Skill] = {s.name: s for s in skills}

    # 创建默认 session
    session = Session(workspace=config.workspace)

    # REPL
    print("Novare 科研智能体 (输入 /help 查看命令)")
    print(f"模型: {config.model} | 工具: {len(tool_registry.list_tools())} 个 | Skills: {len(skills)} 个")
    print("-" * 50)

    try:
        while True:
            try:
                user_input = input("\nnovare> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # 内置命令
            if user_input in ("/exit", "/quit", "/q"):
                break
            elif user_input == "/help":
                _print_help()
                continue
            elif user_input == "/sessions":
                sessions = Session.list_sessions(workspace=config.workspace)
                for s in sessions:
                    marker = " ← current" if s == session.session_id else ""
                    print(f"  {s}{marker}")
                continue
            elif user_input.startswith("/session "):
                sid = user_input.split(" ", 1)[1].strip()
                try:
                    session = Session.load(sid, workspace=config.workspace)
                    print(f"Loaded session: {session.session_id} ({len(session.messages)} messages)")
                except FileNotFoundError:
                    print(f"Session not found: {sid}")
                continue
            elif user_input == "/new":
                session = Session(workspace=config.workspace)
                print(f"New session: {session.session_id}")
                continue
            elif user_input == "/subagents":
                records = subagent_registry.list_all()
                if not records:
                    print("No subagents running or completed.")
                else:
                    print(f"Subagents ({len(records)}):")
                    for r in records:
                        elapsed = f"{r.elapsed:.1f}s"
                        print(f"  {r.subagent_id} [{r.status.value}] {r.type.value} · {elapsed} · {r.task[:50]}")
                continue
            elif user_input == "/skills":
                if not skill_map:
                    print("No skills found. Add .md files to .novare/skills/")
                else:
                    print(f"Available skills ({len(skill_map)}):")
                    for s in skills:
                        desc = f" — {s.description}" if s.description else ""
                        print(f"  /{s.name}{desc}")
                continue
            elif user_input.startswith("/skill "):
                parts = user_input.split(None, 2)
                skill_name = parts[1] if len(parts) > 1 else ""
                skill_args = parts[2] if len(parts) > 2 else ""
                await _invoke_skill(skill_name, skill_args, skill_map, agent, session)
                continue
            elif user_input.startswith("/"):
                print(f"Unknown command: {user_input}. Type /help for available commands.")
                continue

            # 裸词 Skill 匹配：第一个词匹配 skill 名称则自动调用
            first_word = user_input.split()[0] if user_input.split() else ""
            if first_word in skill_map:
                skill_args = user_input[len(first_word):].strip()
                await _invoke_skill(first_word, skill_args, skill_map, agent, session)
                continue

            # 正常对话
            try:
                print()  # 换行
                on_tool = _make_tool_handler()
                result = await agent.run_turn(session, user_input, on_text=lambda t: print(t, end="", flush=True), on_tool=on_tool)
                print()  # 流式结束后换行
                session.save()
            except KeyboardInterrupt:
                print("\n[Interrupted]")
            except Exception as e:
                logger.exception("Error in turn")
                print(f"\nError: {e}")
    finally:
        # 取消所有运行中的子智能体
        cancelled = await subagent_registry.cancel_all()
        if cancelled:
            print(f"\nCancelled {cancelled} running subagent(s).")
        for client in mcp_clients:
            await client.close()
        await llm_client.close()
        session.save()
        print("\n再见！")


def _print_help():
    print("""
Novare 命令:
  /help          显示此帮助
  /skills        列出可用 Skills
  /skill <name>  调用 Skill（也可直接输入 skill 名称）
  /sessions      列出所有会话
  /session <id>  加载指定会话
  /new           创建新会话
  /subagents     列出子智能体状态
  /exit          退出

直接输入文字开始对话，Novare 会自动调用工具完成科研任务。
输入 Skill 名称（如 research Transformer）可快速调用预设流程。
子智能体会在后台自动运行，主智能体可通过 spawn_subagent 创建并行任务。
""")


async def _invoke_skill(
    name: str,
    args: str,
    skill_map: dict[str, Skill],
    agent: AgentLoop,
    session: Session,
):
    """调用一个 skill：渲染模板 → run_turn"""
    skill = skill_map.get(name)
    if not skill:
        print(f"Unknown skill: {name}")
        available = ", ".join(f"/{s}" for s in skill_map) or "(none)"
        print(f"Available: {available}")
        return

    prompt = skill.render(args)
    logger.info("Invoking skill '%s' with args='%s'", name, args)
    print(f"[skill: {name}]")
    try:
        print()
        on_tool = _make_tool_handler()
        result = await agent.run_turn(session, prompt, on_text=lambda t: print(t, end="", flush=True), on_tool=on_tool)
        print()
        session.save()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        logger.exception("Error invoking skill %s", name)
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
