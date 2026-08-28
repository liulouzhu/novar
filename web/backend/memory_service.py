"""用户长期记忆服务 — 自动提取 + system prompt 注入"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING
from uuid import UUID

from web.backend.db.base import get_session_factory
from web.backend.repositories.memory_repo import MemoryRepository

if TYPE_CHECKING:
    from novare.llm_client import LLMClient


# ── 记忆值清洗（防 prompt injection）─────────────────────────

_MAX_VALUE_LEN = 200
_MAX_KEY_LEN = 100
_MAX_TAGS = 10
_MAX_TAG_LEN = 50

_INJECTION_PATTERNS = re.compile(
    r"(忽略[之以]前[的]?指令|ignore previous|ignore all|forget .*instructions"
    r"|你现在是|你是一个|system prompt|reveal .*prompt|输出.*密钥|disregard)",
    re.IGNORECASE,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_memory_value(value: str) -> str:
    """清洗单条记忆 value，防注入 + 截断 + 换行归一化"""
    if not value:
        return ""
    text = value.replace("\r", "").replace("\n", " ")
    text = _CONTROL_CHAR_RE.sub("", text)
    text = text[:_MAX_VALUE_LEN]
    if _INJECTION_PATTERNS.search(text):
        text = "[已标记] " + text
    return text.strip()


def sanitize_memory_key(key: str) -> str:
    """清洗 key：去除控制字符，限制长度。"""
    if not key:
        return ""
    text = _CONTROL_CHAR_RE.sub("", str(key))
    return text[:_MAX_KEY_LEN].strip()


def sanitize_memory_tags(tags: list) -> list[str]:
    """清洗 tags：限制数量、单项长度，去除嵌套对象。"""
    if not tags or not isinstance(tags, list):
        return []
    result = []
    for t in tags[:_MAX_TAGS]:
        if not isinstance(t, (str, int, float)):
            continue
        clean = _CONTROL_CHAR_RE.sub("", str(t))[:_MAX_TAG_LEN].strip()
        if clean:
            result.append(clean)
    return result


def _is_valid_confidence(val) -> bool:
    """验证 confidence 值：有限数字且在 [0, 1] 范围内。"""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return False
        return 0.0 <= f <= 1.0
    except (TypeError, ValueError):
        return False

logger = logging.getLogger("novare.memory")


EXTRACT_MEMORY_PROMPT = """你是一个用户画像分析师。从以下对话中提取用户的偏好信息。

## 对话历史
{conversation}

## 提取规则
1. 只提取**明确表达或强烈暗示**的偏好，不要猜测
2. 每条记忆必须有置信度（0-1）：明确陈述=1.0，强烈暗示=0.8，一般暗示=0.5
3. 如果对话中没有明确的偏好信息，返回空数组 []
4. 不要重复提取已经在"已有记忆"中的信息

## 已有记忆
{existing_memories}

## 输出格式（JSON 数组）
[
  {{
    "category": "research_preference 或 interaction_preference",
    "key": "简短键名，如 research_field",
    "value": "具体值",
    "confidence": 1.0,
    "tags": ["标签1", "标签2"]
  }}
]

## research_preference 可用的 key
- research_field: 研究领域
- research_topics: 感兴趣的具体课题
- preferred_methods: 偏好的研究方法
- familiar_tools: 熟悉的工具/框架
- reading_habits: 阅读习惯（如偏好综述/原始论文）

## interaction_preference 可用的 key
- language: 偏好语言（中文/英文/混合）
- detail_level: 喜欢详细还是简洁的回复
- output_format: 偏好输出格式（表格/列表/段落/代码）
- citation_style: 论文引用偏好

只输出 JSON 数组，不要其他文字。"""


class MemoryService:
    """用户长期记忆服务（同步版本，供非异步上下文使用）"""

    def build_memory_prompt(self, user_id: str) -> str:
        """将用户记忆格式化为 system prompt 片段（带注入防护）"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 已在异步上下文中，无法同步访问异步 Session
            return ""
        except RuntimeError:
            pass
        # 同步路径：不使用异步 DB，返回空（实际提取由 MemoryServiceAsync 完成）
        return ""


class MemoryServiceAsync:
    """异步版本的记忆服务（用于 agent_service 中）"""

    def __init__(self, max_memories: int = 50):
        self.max_memories = max_memories

    # ── 纯持久化入口（供统一提取层调用）───────────────────────

    async def get_extraction_context(self, user_id: str) -> str:
        """获取已有画像文本，供协调层传给 UnifiedMemoryExtractor。

        不调用 LLM，不修改数据。
        """
        return await self._get_existing_text(user_id)

    async def save_extracted(
        self,
        *,
        user_id: str,
        candidates: list,
    ) -> list[dict]:
        """纯持久化入口：保存已提取的用户画像候选。

        负责：category/key/value 校验、sanitize、confidence 校验、
        Repository upsert、auto_pin、max_memories 淘汰、commit/rollback、用户隔离。
        不调用 LLM。

        Raises:
            Exception: 数据库/基础设施失败时抛出异常，不吞掉。
        """
        extracted = []
        for idx, c in enumerate(candidates):
            if isinstance(c, dict):
                item = c
            else:
                item = c.model_dump()
            if not self._validate_memory(item, idx):
                continue
            # 清洗 value/key/tags
            safe_value = sanitize_memory_value(str(item.get("value", "")))
            if not safe_value:
                logger.debug("Skipping profile candidate #%d: value empty after sanitize", idx)
                continue
            safe_key = sanitize_memory_key(str(item.get("key", "")))
            if not safe_key:
                logger.debug("Skipping profile candidate #%d: key empty after sanitize", idx)
                continue
            item["value"] = safe_value
            item["key"] = safe_key
            item["tags"] = sanitize_memory_tags(item.get("tags", []))
            extracted.append(item)
        if not extracted:
            return []
        return await self._save_memories(user_id, extracted)

    # ── 兼容入口（AgentService 不再调用，仅保留兼容）──────────

    async def extract_and_save(
        self,
        user_id: str,
        messages: list[dict],
        llm_client: LLMClient,
    ) -> list[dict]:
        """[兼容入口] 异步提取记忆并保存到数据库。

        新流程使用 MemoryExtractionCoordinator 统一提取，
        此方法保留用于独立单元测试兼容。
        """
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                conversation_parts.append(f"用户: {content[:500]}")
            elif role == "assistant":
                if msg.get("_compacted"):
                    continue
                conversation_parts.append(f"助手: {content[:500]}")

        if len(conversation_parts) < 2:
            return []

        conversation_text = "\n".join(conversation_parts[-20:])

        existing_text = await self._get_existing_text(user_id)

        prompt = EXTRACT_MEMORY_PROMPT.format(
            conversation=conversation_text,
            existing_memories=existing_text or "（暂无）",
        )

        messages_for_llm = [
            {"role": "system", "content": "你是一个用户画像分析师。只输出 JSON 数组，不要其他文字。"},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await llm_client.collect_stream(messages_for_llm)
            content = response.content or ""

            extracted = self._parse_memories(content)
            if not extracted:
                return []

            # 复用统一清洗逻辑
            saved = await self.save_extracted(user_id=user_id, candidates=extracted)
            logger.info("Extracted %d memories for user %s", len(saved), user_id)
            return saved

        except Exception as e:
            logger.warning("Memory extraction failed for user %s: %s", user_id, e)
            return []

    def _parse_memories(self, content: str) -> list[dict]:
        """从 LLM 输出中解析记忆条目"""
        content = content.strip()

        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()

        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [m for m in result if self._validate_memory(m)]
        except json.JSONDecodeError:
            pass

        start = content.find('[')
        end = content.rfind(']')
        if start >= 0 and end > start:
            try:
                result = json.loads(content[start:end + 1])
                if isinstance(result, list):
                    return [m for m in result if self._validate_memory(m)]
            except json.JSONDecodeError:
                pass

        return []

    def _validate_memory(self, item: dict, idx: int = 0) -> bool:
        """验证记忆条目格式（含 confidence 0-1 范围检查）"""
        required = {"category", "key", "value"}
        if not required.issubset(item.keys()):
            logger.debug("Skipping profile candidate #%d: missing required fields", idx)
            return False
        if item["category"] not in ("research_preference", "interaction_preference"):
            logger.debug("Skipping profile candidate #%d: invalid category %s", idx, item.get("category"))
            return False
        conf = item.get("confidence", 1.0)
        if not _is_valid_confidence(conf):
            logger.debug("Skipping profile candidate #%d: invalid confidence %s", idx, conf)
            return False
        return True

    async def _save_memories(self, user_id: str, extracted: list[dict]) -> list[dict]:
        """保存提取的记忆到数据库，超出上限时自动淘汰。

        Raises:
            Exception: 数据库/基础设施失败时抛出异常，由调用方决定是否捕获。
        """
        async with get_session_factory()() as db:
            repo = MemoryRepository(db, UUID(user_id))
            saved = []
            for item in extracted:
                auto_pin = (
                    item["category"] == "research_preference"
                    and float(item.get("confidence", 1.0)) >= 0.8
                )
                memory = await repo.upsert(
                    category=item["category"],
                    key=item["key"],
                    value=item["value"],
                    confidence=float(item.get("confidence", 1.0)),
                    tags=item.get("tags", []),
                    source="auto",
                    pinned=auto_pin,
                )
                saved.append({
                    "category": memory.category,
                    "key": memory.key,
                    "value": memory.value,
                    "pinned": memory.pinned,
                })

            evicted = await repo.evict_excess(self.max_memories)
            if evicted > 0:
                logger.info("Evicted %d excess memories for user %s (max=%d)", evicted, user_id, self.max_memories)

            await db.commit()
            return saved

    async def _get_existing_text(self, user_id: str) -> str:
        """获取已有记忆的文本表示（带注入防护）"""
        try:
            async with get_session_factory()() as db:
                repo = MemoryRepository(db, UUID(user_id))
                memories = await repo.get_all()
        except Exception:
            return ""

        if not memories:
            return ""

        lines = []
        for m in memories:
            safe_value = sanitize_memory_value(m.value)
            lines.append(f"- [{m.category}] {m.key}: {safe_value} (置信度: {m.confidence})")
        return "\n".join(lines)
