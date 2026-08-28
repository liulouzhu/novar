"""统一记忆提取协调层 — 一次 LLM 调用，分别存储画像和情景记忆。

流程：
1. 读取现有画像上下文（通过 MemoryServiceAsync.get_extraction_context）
2. 统一调用一次 LLM（通过 UnifiedMemoryExtractor）
3. 分别保存画像和情景记忆（通过各自 Service 的 save_extracted）

两个存储服务保持失败隔离：一个失败不影响另一个。
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from novare.llm_client import LLMClient
    from web.backend.memory_service import MemoryServiceAsync
    from web.backend.episodic_memory.service import EpisodicMemoryService

from .extractor import UnifiedMemoryExtractor, ExtractionParseError

logger = logging.getLogger("novare.memory_extraction.coordinator")


class ExtractionStatus(str, enum.Enum):
    """记忆提取的结构化状态。"""
    SUCCESS = "success"
    EXTRACTION_FAILED = "extraction_failed"
    PROFILE_PERSIST_FAILED = "profile_persist_failed"
    EPISODIC_PERSIST_FAILED = "episodic_persist_failed"
    BOTH_PERSIST_FAILED = "both_persist_failed"


@dataclass
class ExtractionResult:
    """记忆提取的结构化返回结果。"""
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    profile_saved: int = 0
    episodes_saved: int = 0
    episodes_indexed: int = 0
    episodes_index_failed: int = 0

    @property
    def should_advance_cursor(self) -> bool:
        """是否应推进游标。只有 SUCCESS 才推进。"""
        return self.status == ExtractionStatus.SUCCESS


class MemoryExtractionCoordinator:
    """统一记忆提取协调器。"""

    def __init__(
        self,
        memory_service: MemoryServiceAsync | None = None,
        episodic_memory_service: EpisodicMemoryService | None = None,
    ):
        self._memory_service = memory_service
        self._episodic_memory_service = episodic_memory_service
        self._extractor = UnifiedMemoryExtractor()

    async def extract_and_persist(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: list[dict],
        llm_client: LLMClient,
    ) -> ExtractionResult:
        """统一提取并持久化记忆。

        Returns:
            ExtractionResult with typed status and counts.
        """
        result = ExtractionResult()

        extract_profile = self._memory_service is not None
        extract_episodes = (
            self._episodic_memory_service is not None
            and self._episodic_memory_service.enabled
        )

        # 两者都关闭时直接返回
        if not extract_profile and not extract_episodes:
            return result

        # 读取现有画像上下文
        existing_profile = ""
        if extract_profile:
            try:
                existing_profile = await self._memory_service.get_extraction_context(user_id)
            except Exception:
                logger.debug("Failed to get extraction context (non-fatal)")

        # 计算 max_episodes：从 episodic_memory_service.max_per_turn 获取
        max_episodes = 3
        if extract_episodes:
            max_episodes = self._episodic_memory_service.max_per_turn

        # ── 唯一一次 LLM 调用 ──
        try:
            extraction = await self._extractor.extract(
                messages=messages,
                llm_client=llm_client,
                existing_profile=existing_profile,
                extract_profile=extract_profile,
                extract_episodes=extract_episodes,
                max_episodes=max_episodes,
            )
        except ExtractionParseError:
            logger.warning("Unified memory extraction parse failed for user %s", user_id)
            result.status = ExtractionStatus.EXTRACTION_FAILED
            return result
        except Exception:
            logger.exception("Unified memory extraction failed for user %s", user_id)
            result.status = ExtractionStatus.EXTRACTION_FAILED
            return result

        # ── 分别持久化，失败隔离 ──
        profile_failed = False
        episodic_failed = False

        async def _save_profile():
            nonlocal profile_failed
            if not extract_profile or not extraction.profile_updates:
                return
            try:
                saved = await self._memory_service.save_extracted(
                    user_id=user_id,
                    candidates=extraction.profile_updates,
                )
                result.profile_saved = len(saved)
            except Exception:
                logger.exception("Profile persistence failed for user %s", user_id)
                profile_failed = True

        async def _save_episodes():
            nonlocal episodic_failed
            if not extract_episodes or not extraction.episodes:
                return
            try:
                saved = await self._episodic_memory_service.save_extracted(
                    user_id=user_id,
                    session_id=session_id,
                    candidates=extraction.episodes,
                )
                result.episodes_saved = len(saved)
                # 统计 index_status
                for rec in saved:
                    status = rec.get("index_status", "")
                    if status == "indexed":
                        result.episodes_indexed += 1
                    elif status == "failed":
                        result.episodes_index_failed += 1
            except Exception:
                logger.exception("Episodic persistence failed for user %s", user_id)
                episodic_failed = True

        # 两个存储服务失败隔离
        profile_result, episode_result = await asyncio.gather(
            _save_profile(),
            _save_episodes(),
            return_exceptions=True,
        )

        if isinstance(profile_result, Exception):
            logger.exception("Profile persistence raised for user %s", user_id)
            profile_failed = True
        if isinstance(episode_result, Exception):
            logger.exception("Episodic persistence raised for user %s", user_id)
            episodic_failed = True

        # 确定最终状态
        if profile_failed and episodic_failed:
            result.status = ExtractionStatus.BOTH_PERSIST_FAILED
        elif profile_failed:
            result.status = ExtractionStatus.PROFILE_PERSIST_FAILED
        elif episodic_failed:
            result.status = ExtractionStatus.EPISODIC_PERSIST_FAILED
        else:
            result.status = ExtractionStatus.SUCCESS

        # 结构化日志
        if result.status != ExtractionStatus.SUCCESS:
            logger.warning(
                "Memory extraction completed with status %s for user %s",
                result.status.value,
                user_id,
            )
        else:
            logger.info(
                "Memory extraction completed for user %s: profile_saved=%d, episodes_saved=%d",
                user_id,
                result.profile_saved,
                result.episodes_saved,
            )

        return result
