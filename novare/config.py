"""Novare 配置加载 — 环境变量 + .novare/config.json"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class McpServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def _any_channel_enabled(channels: dict) -> bool:
    """检查 channels 配置中是否有任何渠道 enabled（支持 dict 和 list 格式）。"""
    for ch in channels.values():
        if isinstance(ch, dict) and ch.get("enabled", False):
            return True
        if isinstance(ch, list):
            for item in ch:
                if isinstance(item, dict) and item.get("enabled", False):
                    return True
    return False


@dataclass
class NovareConfig:
    api_key: str = ""
    base_url: str = "https://api.minimax.chat/v1"
    model: str = "MiniMax-Text-01"
    data_dir: Path = Path("./data")
    workspace: Path = Path("./workspace")
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    system_prompt: str = ""
    skill_dirs: list[Path] = field(default_factory=list)

    # 评审模型配置（可选，用于双模型对抗评审）
    # 留空时禁用双模型模式，所有阶段使用主模型
    reviewer_api_key: str = ""
    reviewer_base_url: str = ""
    reviewer_model: str = ""

    # 上下文管理
    auto_compact_threshold: int = 100_000   # 累积 input tokens 超过此值触发自动压缩
    context_max_turns: int = 3              # 最多保留的完整用户轮次
    context_token_budget: int = 12_000       # 压缩后工作上下文软预算
    context_summary_max_tokens: int = 2_500  # LLM 结构化摘要输出上限
    context_tool_result_max_tokens: int = 1_200  # 单个压缩工具结果上限
    context_llm_timeout: int = 30            # 单次压缩 LLM 调用超时
    context_llm_enabled: bool = True         # 失败时自动回退纯规则摘要

    # RAG 回答幻觉检测
    hallucination_verifier_enabled: bool = False
    hallucination_verifier_max_claims: int = 12
    hallucination_verifier_top_k: int = 5
    hallucination_verifier_concurrency: int = 3
    hallucination_verifier_timeout: int = 120

    # 长期记忆
    enable_long_term_memory: bool = True    # 是否启用长期记忆
    max_memories_per_user: int = 50         # 每个用户最大记忆条数

    # 迭代次数限制
    max_iterations: int = 20                # 主 agent 最大工具调用轮次
    subagent_max_iterations: int = 16       # 子智能体最大工具调用轮次

    # 代理
    proxy: str | None = None                # HTTP 代理地址；留空则不使用代理

    # 多渠道接入
    channels_enabled: bool = False              # 是否启用多渠道系统
    channels: dict[str, dict] = field(default_factory=dict)  # 渠道配置，如 {"weixin": {"enabled": True, ...}}
    channel_default_user_id: str = ""           # 渠道消息的默认用户 ID（留空则使用匿名 workspace）

    # Redis（可选，用于分布式锁 / 消息去重等）
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # 超时（秒）
    turn_timeout: int = 300                 # 主 agent 单轮超时（默认 5 分钟）
    subagent_turn_timeout: int = 600        # 子智能体单轮超时（默认 10 分钟）

    # Agent 运行时（LangGraph 重构：legacy / langgraph 双运行时并存）
    agent_runtime: str = "legacy"           # "legacy" | "langgraph"
    model_port: str = "legacy"              # "legacy" | "langchain"（graph 模式的模型适配）

    # 重试（PR 1：统一错误模型与基础重试机制）
    llm_retry_attempts: int = 3             # LLM 调用最多尝试次数（含首次）
    retry_base_delay: float = 0.5           # 指数退避基数（秒）
    retry_max_delay: float = 8.0            # 指数退避上限（秒）
    max_retries_per_turn: int = 6           # 每轮 LLM + 工具共享的重试预算
    retry_after_max_delay: float = 30.0     # 服务端 Retry-After 上限（秒）

    # Reflexion（PR 3：确定性触发 + 结构化策略修订）
    reflexion_enabled: bool = False          # 主 Agent 默认关闭，开启后行为才变化
    max_reflections_per_turn: int = 2       # 每轮最多反思次数
    reflexion_no_progress_threshold: int = 3     # 连续无进展 N 轮触发反思
    reflexion_repeated_failure_threshold: int = 2  # 相同动作连续失败 N 次触发反思
    reflexion_timeout: float = 30.0         # 单次反思 LLM 调用超时（秒）
    reflexion_max_tokens: int = 1200        # 反思输出上限
    reflexion_max_recent_events: int = 8    # 反思输入最多携带的最近事件数

    # 情景记忆（Episodic Memory）
    episodic_memory_enabled: bool = False   # 默认关闭，向后兼容
    episodic_memory_top_k: int = 5          # 检索返回条数
    episodic_memory_min_importance: float = 0.6   # 最低重要性阈值
    episodic_memory_min_confidence: float = 0.7   # 最低置信度阈值
    episodic_memory_min_similarity: float = 0.55  # 最低语义相似度阈值
    episodic_memory_max_per_turn: int = 3         # 每轮最多保存条数
    episodic_memory_collection: str = "episodic_memories"  # Milvus collection 名

    # 批量记忆提取调度
    memory_extraction_interval_turns: int = 4      # 累计多少个完整轮次后提取一次（1-50）
    memory_extraction_idle_seconds: int = 120      # 空闲多少秒后 flush（10-3600）
    memory_extraction_flush_on_switch: bool = True  # 切换会话时是否 flush 旧会话

    # 测试 embedding fallback（生产环境禁止启用）
    test_embedding_fallback: bool = False

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "NovareConfig":
        """从 .env 文件、环境变量和配置文件加载配置"""
        cfg = cls()

        # 加载 .env 文件（当前工作目录）
        load_dotenv(Path.cwd() / ".env")

        # 环境变量覆盖
        cfg.api_key = os.environ.get("NOVARE_API_KEY", cfg.api_key)
        cfg.base_url = os.environ.get("NOVARE_BASE_URL", cfg.base_url)
        cfg.model = os.environ.get("NOVARE_MODEL", cfg.model)

        # 评审模型（可选）
        cfg.reviewer_api_key = os.environ.get("NOVARE_REVIEWER_API_KEY", cfg.reviewer_api_key)
        cfg.reviewer_base_url = os.environ.get("NOVARE_REVIEWER_BASE_URL", cfg.reviewer_base_url)
        cfg.reviewer_model = os.environ.get("NOVARE_REVIEWER_MODEL", cfg.reviewer_model)

        data_dir = os.environ.get("NOVARE_DATA_DIR")
        if data_dir:
            cfg.data_dir = Path(data_dir).resolve()

        workspace = os.environ.get("NOVARE_WORKSPACE")
        if workspace:
            cfg.workspace = Path(workspace).resolve()

        # 迭代次数
        max_iter = os.environ.get("NOVARE_MAX_ITERATIONS")
        if max_iter:
            cfg.max_iterations = int(max_iter)
        sub_max_iter = os.environ.get("NOVARE_SUBAGENT_MAX_ITERATIONS")
        if sub_max_iter:
            cfg.subagent_max_iterations = int(sub_max_iter)

        turn_timeout = os.environ.get("NOVARE_TURN_TIMEOUT")
        if turn_timeout:
            cfg.turn_timeout = int(turn_timeout)
        sub_turn_timeout = os.environ.get("NOVARE_SUBAGENT_TURN_TIMEOUT")
        if sub_turn_timeout:
            cfg.subagent_turn_timeout = int(sub_turn_timeout)

        # Agent 运行时（LangGraph 重构）
        agent_runtime = os.environ.get("NOVARE_AGENT_RUNTIME")
        if agent_runtime and agent_runtime in ("legacy", "langgraph"):
            cfg.agent_runtime = agent_runtime
        model_port = os.environ.get("NOVARE_MODEL_PORT")
        if model_port and model_port in ("legacy", "langchain"):
            cfg.model_port = model_port

        # 重试配置（PR 1）
        llm_retry = os.environ.get("NOVARE_LLM_RETRY_ATTEMPTS")
        if llm_retry:
            cfg.llm_retry_attempts = int(llm_retry)
        retry_base = os.environ.get("NOVARE_RETRY_BASE_DELAY")
        if retry_base:
            cfg.retry_base_delay = float(retry_base)
        retry_max = os.environ.get("NOVARE_RETRY_MAX_DELAY")
        if retry_max:
            cfg.retry_max_delay = float(retry_max)
        max_retries_turn = os.environ.get("NOVARE_MAX_RETRIES_PER_TURN")
        if max_retries_turn:
            cfg.max_retries_per_turn = int(max_retries_turn)
        retry_after_max = os.environ.get("NOVARE_RETRY_AFTER_MAX_DELAY")
        if retry_after_max:
            cfg.retry_after_max_delay = float(retry_after_max)

        # Reflexion 配置（PR 3）
        reflexion_enabled = os.environ.get("NOVARE_REFLEXION_ENABLED")
        if reflexion_enabled is not None:
            cfg.reflexion_enabled = reflexion_enabled.lower() in ("1", "true", "yes")
        max_reflections = os.environ.get("NOVARE_MAX_REFLECTIONS_PER_TURN")
        if max_reflections:
            cfg.max_reflections_per_turn = int(max_reflections)
        refl_np = os.environ.get("NOVARE_REFLECTION_NO_PROGRESS_THRESHOLD")
        if refl_np:
            cfg.reflexion_no_progress_threshold = int(refl_np)
        refl_rf = os.environ.get("NOVARE_REFLECTION_REPEATED_FAILURE_THRESHOLD")
        if refl_rf:
            cfg.reflexion_repeated_failure_threshold = int(refl_rf)
        refl_timeout = os.environ.get("NOVARE_REFLECTION_TIMEOUT")
        if refl_timeout:
            cfg.reflexion_timeout = float(refl_timeout)
        refl_tokens = os.environ.get("NOVARE_REFLECTION_MAX_TOKENS")
        if refl_tokens:
            cfg.reflexion_max_tokens = int(refl_tokens)
        refl_events = os.environ.get("NOVARE_REFLECTION_MAX_RECENT_EVENTS")
        if refl_events:
            cfg.reflexion_max_recent_events = int(refl_events)

        # 代理
        proxy = os.environ.get("NOVARE_PROXY")
        if proxy:
            cfg.proxy = proxy

        # Redis（可选）
        redis_enabled = os.environ.get("NOVARE_REDIS_ENABLED")
        if redis_enabled is not None:
            cfg.redis_enabled = redis_enabled.lower() in ("1", "true", "yes")
        redis_url = os.environ.get("NOVARE_REDIS_URL")
        if redis_url:
            cfg.redis_url = redis_url

        # 多渠道（环境变量覆盖）
        channels_enabled = os.environ.get("NOVARE_CHANNELS_ENABLED")
        if channels_enabled:
            cfg.channels_enabled = channels_enabled.lower() in ("1", "true", "yes")
        channel_default_user = os.environ.get("NOVARE_CHANNEL_DEFAULT_USER_ID")
        if channel_default_user:
            cfg.channel_default_user_id = channel_default_user
        weixin_token = os.environ.get("NOVARE_WEIXIN_TOKEN")
        if weixin_token:
            cfg.channels.setdefault("weixin", {})["token"] = weixin_token
            cfg.channels["weixin"]["enabled"] = True
            cfg.channels_enabled = True

        # 长期记忆
        enable_memory = os.environ.get("NOVARE_ENABLE_LONG_TERM_MEMORY")
        if enable_memory is not None:
            cfg.enable_long_term_memory = enable_memory.lower() in ("1", "true", "yes")
        max_mem = os.environ.get("NOVARE_MAX_MEMORIES_PER_USER")
        if max_mem:
            cfg.max_memories_per_user = int(max_mem)

        # 上下文管理
        compact_threshold = os.environ.get("NOVARE_AUTO_COMPACT_THRESHOLD")
        if compact_threshold:
            cfg.auto_compact_threshold = int(compact_threshold)
        context_max_turns = os.environ.get("NOVARE_CONTEXT_MAX_TURNS")
        if context_max_turns:
            cfg.context_max_turns = int(context_max_turns)
        context_token_budget = os.environ.get("NOVARE_CONTEXT_TOKEN_BUDGET")
        if context_token_budget:
            cfg.context_token_budget = int(context_token_budget)
        context_summary_max = os.environ.get("NOVARE_CONTEXT_SUMMARY_MAX_TOKENS")
        if context_summary_max:
            cfg.context_summary_max_tokens = int(context_summary_max)
        context_tool_max = os.environ.get("NOVARE_CONTEXT_TOOL_RESULT_MAX_TOKENS")
        if context_tool_max:
            cfg.context_tool_result_max_tokens = int(context_tool_max)
        context_llm_timeout = os.environ.get("NOVARE_CONTEXT_LLM_TIMEOUT")
        if context_llm_timeout:
            cfg.context_llm_timeout = int(context_llm_timeout)
        context_llm_enabled = os.environ.get("NOVARE_CONTEXT_LLM_ENABLED")
        if context_llm_enabled is not None:
            cfg.context_llm_enabled = context_llm_enabled.lower() in ("1", "true", "yes")

        verifier_enabled = os.environ.get("NOVARE_HALLUCINATION_VERIFIER_ENABLED")
        if verifier_enabled is not None:
            cfg.hallucination_verifier_enabled = verifier_enabled.lower() in ("1", "true", "yes")
        verifier_max_claims = os.environ.get("NOVARE_HALLUCINATION_VERIFIER_MAX_CLAIMS")
        if verifier_max_claims:
            cfg.hallucination_verifier_max_claims = int(verifier_max_claims)
        verifier_top_k = os.environ.get("NOVARE_HALLUCINATION_VERIFIER_TOP_K")
        if verifier_top_k:
            cfg.hallucination_verifier_top_k = int(verifier_top_k)
        verifier_concurrency = os.environ.get("NOVARE_HALLUCINATION_VERIFIER_CONCURRENCY")
        if verifier_concurrency:
            cfg.hallucination_verifier_concurrency = int(verifier_concurrency)
        verifier_timeout = os.environ.get("NOVARE_HALLUCINATION_VERIFIER_TIMEOUT")
        if verifier_timeout:
            cfg.hallucination_verifier_timeout = int(verifier_timeout)

        # 情景记忆
        ep_enabled = os.environ.get("NOVARE_EPISODIC_MEMORY_ENABLED")
        if ep_enabled is not None:
            cfg.episodic_memory_enabled = ep_enabled.lower() in ("1", "true", "yes")
        ep_top_k = os.environ.get("NOVARE_EPISODIC_MEMORY_TOP_K")
        if ep_top_k:
            cfg.episodic_memory_top_k = int(ep_top_k)
        ep_min_imp = os.environ.get("NOVARE_EPISODIC_MEMORY_MIN_IMPORTANCE")
        if ep_min_imp:
            cfg.episodic_memory_min_importance = float(ep_min_imp)
        ep_min_conf = os.environ.get("NOVARE_EPISODIC_MEMORY_MIN_CONFIDENCE")
        if ep_min_conf:
            cfg.episodic_memory_min_confidence = float(ep_min_conf)
        ep_max_turn = os.environ.get("NOVARE_EPISODIC_MEMORY_MAX_PER_TURN")
        if ep_max_turn:
            cfg.episodic_memory_max_per_turn = int(ep_max_turn)
        ep_collection = os.environ.get("NOVARE_EPISODIC_MEMORY_COLLECTION")
        if ep_collection:
            cfg.episodic_memory_collection = ep_collection
        ep_min_sim = os.environ.get("NOVARE_EPISODIC_MEMORY_MIN_SIMILARITY")
        if ep_min_sim:
            cfg.episodic_memory_min_similarity = float(ep_min_sim)

        # 批量记忆提取调度
        mem_interval = os.environ.get("NOVARE_MEMORY_EXTRACTION_INTERVAL_TURNS")
        if mem_interval:
            cfg.memory_extraction_interval_turns = int(mem_interval)
        mem_idle = os.environ.get("NOVARE_MEMORY_EXTRACTION_IDLE_SECONDS")
        if mem_idle:
            cfg.memory_extraction_idle_seconds = int(mem_idle)
        mem_flush = os.environ.get("NOVARE_MEMORY_EXTRACTION_FLUSH_ON_SWITCH")
        if mem_flush is not None:
            cfg.memory_extraction_flush_on_switch = mem_flush.lower() in ("1", "true", "yes")

        # 测试 embedding fallback
        test_fallback = os.environ.get("NOVARE_TEST_EMBEDDING_FALLBACK")
        if test_fallback is not None:
            cfg.test_embedding_fallback = test_fallback.lower() in ("1", "true", "yes")

        # 情景记忆配置校验
        cfg.episodic_memory_top_k = max(1, min(20, cfg.episodic_memory_top_k))
        cfg.episodic_memory_min_importance = max(0.0, min(1.0, cfg.episodic_memory_min_importance))
        cfg.episodic_memory_min_confidence = max(0.0, min(1.0, cfg.episodic_memory_min_confidence))
        cfg.episodic_memory_min_similarity = max(-1.0, min(1.0, cfg.episodic_memory_min_similarity))
        cfg.episodic_memory_max_per_turn = max(1, min(10, cfg.episodic_memory_max_per_turn))

        # 上下文压缩配置校验
        cfg.context_max_turns = max(1, min(20, cfg.context_max_turns))
        cfg.context_token_budget = max(1_000, min(200_000, cfg.context_token_budget))
        cfg.context_summary_max_tokens = max(300, min(8_000, cfg.context_summary_max_tokens))
        cfg.context_tool_result_max_tokens = max(200, min(5_000, cfg.context_tool_result_max_tokens))
        cfg.context_llm_timeout = max(1, min(120, cfg.context_llm_timeout))

        # 重试配置校验（PR 1）
        cfg.llm_retry_attempts = max(1, min(10, cfg.llm_retry_attempts))
        cfg.retry_base_delay = max(0.0, min(60.0, cfg.retry_base_delay))
        cfg.retry_max_delay = max(cfg.retry_base_delay, min(600.0, cfg.retry_max_delay))
        cfg.max_retries_per_turn = max(0, min(100, cfg.max_retries_per_turn))
        cfg.retry_after_max_delay = max(1.0, min(600.0, cfg.retry_after_max_delay))

        # Reflexion 配置校验（PR 3）
        cfg.max_reflections_per_turn = max(1, min(10, cfg.max_reflections_per_turn))
        cfg.reflexion_no_progress_threshold = max(1, min(20, cfg.reflexion_no_progress_threshold))
        cfg.reflexion_repeated_failure_threshold = max(1, min(20, cfg.reflexion_repeated_failure_threshold))
        cfg.reflexion_timeout = max(1.0, min(300.0, cfg.reflexion_timeout))
        cfg.reflexion_max_tokens = max(100, min(8000, cfg.reflexion_max_tokens))
        cfg.reflexion_max_recent_events = max(1, min(50, cfg.reflexion_max_recent_events))

        # 幻觉检测配置校验
        cfg.hallucination_verifier_max_claims = max(
            1, min(15, cfg.hallucination_verifier_max_claims)
        )
        cfg.hallucination_verifier_top_k = max(
            1, min(10, cfg.hallucination_verifier_top_k)
        )
        cfg.hallucination_verifier_concurrency = max(
            1, min(8, cfg.hallucination_verifier_concurrency)
        )
        cfg.hallucination_verifier_timeout = max(
            10, min(600, cfg.hallucination_verifier_timeout)
        )

        # 批量记忆提取配置校验
        cfg.memory_extraction_interval_turns = max(1, min(50, cfg.memory_extraction_interval_turns))
        cfg.memory_extraction_idle_seconds = max(10, min(3600, cfg.memory_extraction_idle_seconds))
        # Collection 名只允许安全字符
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', cfg.episodic_memory_collection):
            raise ValueError(
                f"Invalid episodic_memory_collection name: {cfg.episodic_memory_collection!r}. "
                "Only alphanumeric characters and underscores are allowed."
            )
        path = Path(config_path).resolve() if config_path else cfg.workspace / ".novare" / "config.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "mcpServers" in data:
                    for name, srv in data["mcpServers"].items():
                        if isinstance(srv, dict) and "command" in srv:
                            cfg.mcp_servers[name] = McpServerConfig(
                                command=srv["command"],
                                args=srv.get("args", []),
                                env=srv.get("env", {}),
                            )
                # 多渠道配置
                if isinstance(data, dict) and "channels" in data:
                    cfg.channels = data["channels"]
                    cfg.channels_enabled = _any_channel_enabled(cfg.channels)
            except (json.JSONDecodeError, OSError):
                pass  # 配置文件损坏时使用默认值

        # 默认研究工具 MCP 服务器（在项目根目录查找）
        project_root = Path.cwd()
        if not cfg.mcp_servers and (project_root / "mcp-server").exists():
            venv_python = project_root / ".venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = project_root / ".venv" / "bin" / "python"
            mcp_server_py = project_root / "mcp-server" / "research_server.py"
            if venv_python.exists() and mcp_server_py.exists():
                # 显式 allowlist：只传递 MCP 子进程所需的环境变量
                mcp_env: dict[str, str] = {
                    "RESEARCH_DATA_DIR": str(cfg.data_dir),
                    "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
                }
                # Embedding 配置
                for key in ("DASHSCOPE_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL"):
                    val = os.environ.get(key)
                    if val:
                        mcp_env[key] = val
                # Milvus 配置
                for key in ("MILVUS_HOST", "MILVUS_PORT"):
                    val = os.environ.get(key)
                    if val:
                        mcp_env[key] = val
                # RAG 配置
                for key in ("RAG_DEFAULT_USER", "RAG_ALLOW_UNSCOPED", "NOVARE_TEST_EMBEDDING_FALLBACK"):
                    val = os.environ.get(key)
                    if val:
                        mcp_env[key] = val
                # Elasticsearch 配置
                for key in ("ELASTICSEARCH_URL", "ELASTICSEARCH_INDEX",
                            "ELASTICSEARCH_USERNAME", "ELASTICSEARCH_PASSWORD",
                            "RAG_VECTOR_TOP_N", "RAG_KEYWORD_TOP_N", "RAG_RRF_K",
                            "RAG_RERANK_ENABLED", "RAG_RERANK_API_KEY",
                            "RAG_RERANK_URL", "RAG_RERANK_MODEL",
                            "RAG_RERANK_CANDIDATES", "RAG_RERANK_TIMEOUT",
                            "RAG_RERANK_MAX_DOC_CHARS"):
                    val = os.environ.get(key)
                    if val:
                        mcp_env[key] = val
                cfg.mcp_servers["research"] = McpServerConfig(
                    command=str(venv_python),
                    args=[str(mcp_server_py)],
                    env=mcp_env,
                )

        # 默认系统提示词
        if not cfg.system_prompt:
            cfg.system_prompt = _default_system_prompt(cfg.workspace)

        # Skill 目录：系统公共 + 用户私有（~/.novare/skills 兜底）
        skill_dirs_env = os.environ.get("NOVARE_SKILL_DIR")
        if skill_dirs_env:
            cfg.skill_dirs = [Path(p).resolve() for p in skill_dirs_env.split(os.pathsep)]
        else:
            cfg.skill_dirs = [
                project_root / "system" / "skills",
                Path.home() / ".novare" / "skills",
            ]

        return cfg


def get_user_workspace(user_id: str) -> str:
    """Return isolated workspace path for a user."""
    import os
    base = os.getenv("NOVARE_WORKSPACE", "workspace")
    path = os.path.join(base, user_id)
    os.makedirs(os.path.join(path, ".novare", "sessions"), exist_ok=True)
    os.makedirs(os.path.join(path, "uploads"), exist_ok=True)
    return path


def _default_system_prompt(workspace: Path) -> str:
    return f"""你是 Novare 智能科研助手。

你的能力：
- 搜索学术论文（paper_search）
- 解析论文 PDF（paper_parse）
- 在已解析论文中语义检索（rag_query）
- 查询/更新论文知识图谱（knowledge_graph）
- 执行 Python 代码做数据分析（code_execute）
- 读写文件（read_file, write_file, edit_file）
- 搜索文件（glob_search, grep_search）

如果评审模型已配置，你还拥有：
- reviewer_evaluate：用独立的评审模型对候选创新点做对抗评审（双模型模式）

工作空间：{workspace}

工作流指引：
1. 先搜索获取论文列表和 ID
2. 用 paper_parse 解析感兴趣的论文 PDF（解析完成后会自动从摘要提取实体到知识图谱）
3. 用 rag_query 在已解析的论文库中语义问答
4. 用 knowledge_graph 构建概念关系图谱（查询自动构建的实体，或手动补充更多关系）
5. 用 code_execute 进行数据分析和可视化

知识图谱指引：
- paper_parse 解析完成后会自动提取摘要中的方法、数据集、任务实体
- 可以用 knowledge_graph(action="query") 查看已有实体和关系
- 可以用 knowledge_graph(action="extract_from_abstract", paper_id="...", entities=[...]) 手动补充实体
- 可以用 knowledge_graph(action="find_path") 发现概念之间的关联路径

输出规范：
- 引用论文时提供标题、作者、年份
- 综述回答按主题组织，引用具体论文
- 区分已解析论文（有全文）和仅检索到的论文（仅有摘要）
- 使用中文与用户交互，搜索词建议使用英文以获得更好的检索效果

任务状态指引：
- 你会看到 [当前任务状态] 块，包含目标、已完成步骤、待办步骤、关键发现
- 每次工具调用后审视状态：信息是否足够回答用户？是否有明显缺失？
- 如果关键发现已覆盖用户问题的核心维度，直接综合回答，不要过度搜索
- 如果发现缺失重要信息（如只找到 1-2 篇论文但用户要求综述），继续检索
- 不要重复已经完成的步骤（如已经搜索过的查询词）
"""
