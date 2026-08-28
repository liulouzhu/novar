"""tests/test_memory_service.py — MemoryServiceAsync 长期记忆测试

验证：
- extract_and_save 正确 await _get_existing_text
- 已有记忆文本进入 LLM Prompt
- 不存在未 await 的协程
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_extract_and_save_awaits_get_existing_text():
    """extract_and_save 正确 await _get_existing_text。"""
    from web.backend.memory_service import MemoryServiceAsync

    service = MemoryServiceAsync(max_memories=50)
    service._get_existing_text = AsyncMock(return_value="- [research_preference] field: NLP (置信度: 0.9)")

    with patch.object(service, "_save_memories", new_callable=AsyncMock, return_value=[]):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_llm.collect_stream = AsyncMock(return_value=mock_response)

        await service.extract_and_save("user-1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ], mock_llm)

    # _get_existing_text 应该被 await
    service._get_existing_text.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_existing_memory_text_enters_prompt():
    """已有记忆文本实际进入 LLM Prompt。"""
    from web.backend.memory_service import MemoryServiceAsync

    service = MemoryServiceAsync(max_memories=50)
    service._get_existing_text = AsyncMock(return_value="- [research_preference] field: NLP (置信度: 0.9)")

    with patch.object(service, "_save_memories", new_callable=AsyncMock, return_value=[]):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_llm.collect_stream = AsyncMock(return_value=mock_response)

        await service.extract_and_save("user-1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ], mock_llm)

    # 检查 LLM 调用时的 prompt 中包含已有记忆
    call_args = mock_llm.collect_stream.call_args[0][0]
    user_content = call_args[1]["content"]
    assert "NLP" in user_content
    assert "已有记忆" in user_content or "existing_memories" in user_content or "暂无" not in user_content
