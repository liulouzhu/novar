# Novare LangGraph 重构进度（工作区 D:\Agent\novar）

本工作区是 `D:\project\research-agent` 的重构副本。对应重构计划
（原项目 `LANGGRAPH_REFACTOR_PLAN.md`）的阶段 0 + 阶段 1（MVP）已完成，
"必须重构"的四个模块全部落地。

## 已完成（阶段 0 + 1 MVP）

### 1. 新增 `novare/graph/` — LangGraph 运行时

| 文件 | 职责 |
| --- | --- |
| `state.py` | `GraphState` TypedDict（messages/task_state/rag_used/run_status 等，全部可序列化）+ `merge_messages` reducer + `ReplaceMessages` 全量替换哨兵 |
| `context.py` | `RunContext`（turn-scoped 运行依赖：model/tools/session/回调/RecoveryState/RetryBudget）+ `RuntimeOptions`；不进 checkpoint 序列化 |
| `nodes/model.py` | `call_model`（流式 + 传输层重试 + usage 记账 + pending_tool_calls 写入 state）+ preflight/post-turn 压缩 |
| `nodes/tools.py` | `execute_tools`（retry_tool_call + 幂等键注入 + commit_tool_result_once + 完整性检查 + task_state 更新 + rag_used 检测）+ `route_after_tools` |
| `nodes/finalize.py` | `bootstrap` / `verify_answer`（幻觉校验）/ `finalize`（三路终态收敛）+ `route_after_model` |
| `builder.py` | `build_graph(ctx)`：START→bootstrap→call_model→(tools|verify|finalize)→END，每 turn 闭包构图（MVP 无 checkpointer） |
| `runner.py` | `GraphRunner.run_turn()` 与 `AgentLoop.run_turn` 签名完全一致；timeout/cancel/exception 三路 terminalize |

### 2. 模型适配层 `novare/graph/adapters/model.py`

- `ModelPort` 协议：`astream_collect(messages, tools, on_delta) -> ModelResult`
- `LegacyModelPort`：包装现有 `LLMClient`（保留 reasoning_content）
- `LangChainModelPort`：包装 `BaseChatModel`（ChatOpenAI），流式解析 / tool_call
  增量聚合 / usage 转换全部交给 LangChain
- `ChatCompatLLM`：给 HybridContextCompactor 提供最小 `.chat()` 兼容
- 由 `NOVARE_MODEL_PORT=legacy|langchain` 选择

### 3. `novare/tools/registry.py` 重构

- `timeout_seconds` 从死配置变为实际生效（`asyncio.wait_for`）
- `execute()` 异常不再吞成纯文本，返回结构化错误 JSON
  （`ok/error/error_code/retryable/outcome/attempts`），与
  `recovery.classifier` 的结构化字段优先分类直接对齐
- 新增 `to_structured_tool()` / `to_structured_tools()`（ToolDef → LangChain StructuredTool）
- 未知工具 / 无 handler 同样返回结构化错误（UNKNOWN_TOOL / NO_HANDLER）

### 4. `web/backend/agent_service.py` 接入

- `_build_agent_runtime()`：按 `NOVARE_AGENT_RUNTIME` 分发 AgentLoop / GraphRunner
- 两个运行时 `run_turn` 签名兼容，调用点零改动（`self.agent.run_turn(...)`）
- Reflexion 未迁移，graph 模式 + reflexion_enabled 自动回落 legacy 并告警
- 顺手修复了原文件 `self.agent = AgentLoop(            llm_client=...` 的排版损坏

### 5. 其他修改

- `novare/config.py`：新增 `agent_runtime` / `model_port` 字段与 env 读取
- `novare/recovery/state.py`：`commit_tool_result_once` 从 agent_loop.py 移入
  （协议层归属），agent_loop.py 保留 re-export 兼容旧 import 路径
- `novare/session.py`：修复 "latest" 解析的 mtime 同秒抖动（文件名时间戳兜底排序）

## 测试

- `tests/test_graph_runtime.py`（17 个）：纯文本 / 工具循环 / max_iterations /
  取消 / RAG 校验 / usage / RecoveryState 事件 / 结构化错误 / timeout /
  双 ModelPort / reducer
- 全量回归：865 passed, 2 skipped（含 legacy 全部测试）

## 待办（后续阶段）

- 阶段 2：`langgraph-checkpoint-postgres`，thread_id = user_id:session_id，
  取代 RecoveryState 快照持久化（并消除 agent_service 两个回调写同一 DB 行的竞态）
- 阶段 3：Reflexion 子图（triggers/progress 纯函数保留）、子 agent Send API、
  统一三套 LLM JSON 解析（compactor/verifier/reflexion → with_structured_output）
- 阶段 5：CLI / channels 切到 GraphRunner，删除 legacy AgentLoop

## 复制的目录

`novare/`、`web/backend/`、`tests/`、`mcp-server/`（除 data）、`system/`、
`docker/sandbox/`、`scripts/`、`alembic.ini`、`.env.example`。
未复制：前端、benchmarks、uv-cache、原 venv。
