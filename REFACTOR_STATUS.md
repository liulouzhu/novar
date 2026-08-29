# Novare LangGraph 重构进度（工作区 D:\Agent\novar）

本工作区是 `D:\project\research-agent` 的重构副本。对应重构计划
（原项目 `LANGGRAPH_REFACTOR_PLAN.md`）：阶段 0 + 1（MVP）、
"必须重构"四模块、"建议重构"六项、**阶段 2（checkpoint 持久化与恢复）** 已完成。

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

## 第二批：建议重构（保留实现，改造接入方式）

### 6. 统一三套 LLM JSON 解析（新增 `novare/llm_json.py`）

- `parse_json_object(text, tolerant=)`：严格模式（剥围栏 + 恰好一个对象）
  与容忍模式（提取 `{...}` 块 / 返回 None）统一实现
- `call_llm_json(...)`：`llm_client.chat()` + 错误反馈重试的统一循环，
  `validate`（只校验）/ `convert`（校验后转换）两个显式参数；空内容诊断
  （finish_reason / reasoning_chars）内建
- 三处替换：verifier `_call_json`（30 行循环 → 薄包装）、compactor
  `_summarize_with_llm`（重试循环 → convert 回调）、reflexion
  `parse_model_json`（→ 容忍模式 re-export）。三份逐行重复的
  `_parse_json_object` 全部删除，verifier 保留 re-export 兼容测试 import

### 7. recovery 收敛

- 新增 `query_tool_retry_semantics(registry, name)`：retry/idempotency 的
  getattr 探测逻辑收敛为唯一实现，`agent_loop._tool_retry_policy`、
  `RecoveryState.register_tool_calls_batch`、graph `nodes/tools._tool_retry_policy`
  三处统一调用
- `compute_action_fingerprint` 提升为 state.py 公开 API，
  `reflexion/triggers.py` 不再引用私有 `_compute_action_fingerprint`
- 评估后未做 to_dict 快照 memo：快照规模小（每 batch 1-3 个 tool call），
  脏标记需覆盖全部 mutator、漏标会导致恢复状态陈旧，风险大于收益

### 8. reflexion / task_state / subagents / config

- `reflexion/prompts.py:77` 排版损坏修复（注释与 dict 闭合挤一行）
- `task_state.py`：if/elif 工具分发改为模块级 `_TOOL_HANDLERS` 映射表 +
  `register_tool_handler()` 公开 API —— MCP 动态工具可外部注册提取器
- `subagents/tools.py`：`await_result` 的硬编码 300s 超时参数化
  （`subagent_await_timeout`，默认与子 agent `turn_timeout` 对齐，
  消除 300s < 600s 提前放弃的矛盾）
- `config.py`：63 处 env 样板收敛为 `_env_str/_env_int/_env_float/_env_bool/_env_path`
  辅助（clamp 边界内联保留、retry_max_delay 的耦合 clamp 保留、布尔的
  `is not None` 语义保留、非法值告警回退）；`McpServerConfig` 双定义统一到
  config.py（补 `cwd` 字段），`mcp_client.py` 改为 re-export

### 测试

- 新增 `tests/test_llm_json.py`（16 个）：严格/容忍解析、错误反馈重试、
  validate/convert 语义、空内容诊断、handler 注册、超时参数化
- 全量：881 passed, 2 skipped

## 第三批：阶段 2 — checkpoint 持久化与恢复

### 数据职责（按计划文档执行）

| 数据 | 唯一数据源 |
| --- | --- |
| 用户可见消息 | PostgreSQL 消息表（不变） |
| 会话标题和列表 | Session Repository（不变） |
| Graph 执行现场 | LangGraph checkpoint（新增） |
| 工具副作用和幂等记录 | RecoveryState / recovery 表（保留） |
| 任务锁和取消状态 | Redis（不变） |

### 实现

- **`novare/graph/checkpointer.py`（新）**：`build_checkpointer(DATABASE_URL)`
  工厂 —— PostgreSQL → AsyncPostgresSaver（自动建表，+asyncpg URL 自动转换）；
  无数据库 → InMemorySaver（测试/单进程回退）。`build_thread_id()`：
  `user_id:session_id:run_id`（**turn 级**执行现场，与 RecoveryState 生命周期
  对齐；不采用会话级累积，避免与消息表双数据源漂移）
- **静态编译图**：`build_graph()` 改为无 ctx 构造，GraphRunner 构造时编译一次
  挂载 checkpointer；per-turn 依赖经 `config["configurable"]["run_ctx"]` 注入
  （不可序列化对象不进 checkpoint）；条件边改纯 state 函数
  （`iteration_limit` / `verify_enabled` 在 turn 开始写入 state），
  保证恢复后的路由决策与首次执行一致
- **恢复对账**（`nodes/tools.py::_reconcile_with_ledger`）：resume 重入
  execute_tools 时按副作用账本处置 —— COMPLETED/FAILED/已终态 → 跳过；
  PENDING/EXECUTING + 非幂等 → 合成 UNKNOWN_OUTCOME 交模型决定；
  PENDING/EXECUTING + 幂等 → 允许重放。仅 `resumed=True` 时生效
- **thread_id 映射**：RecoveryState 增加 `thread_id` 字段（向后兼容），
  随快照自动持久化到 recovery_states 表
- **恢复路径**（agent_service）：`_restore_recovery_ledger()` fail-closed
  恢复账本；graph 运行时显式 resume 传 `thread_id + resume + recovery_state`
  走 checkpoint 恢复；checkpointer 构建失败自动回落 legacy
- **checkpoint 清理**：正常完成（completed）的 turn 自动 `adelete_thread`；
  cancelled/timeout/error/max_iterations 保留供显式恢复

### 阶段 2 测试

- 新增 `tests/test_graph_checkpoint.py`（7 个）：thread_id 映射、中途崩溃
  resume 后工具零重放（副作用恰好一次）、completed 清理、对账三路处置
- 全量：888 passed, 2 skipped
