"""novare/llm_client.py — OpenAI 兼容 API 客户端（支持流式输出）"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

logger = logging.getLogger("novare.llm")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict = field(default_factory=dict)
    # Some OpenAI-compatible reasoning models expose thinking separately from
    # the final answer. Consumers should use this only for diagnostics.
    reasoning_content: str = ""


@dataclass
class StreamChunk:
    """流式输出的单个 chunk"""
    content_delta: str = ""          # 文本增量
    reasoning_delta: str = ""        # 推理过程增量（MiMo 等推理模型）
    tool_call_index: int | None = None
    tool_call_id: str = ""
    tool_call_name: str = ""
    tool_call_arguments_delta: str = ""  # JSON 字符串增量
    finish_reason: str | None = None
    usage: dict | None = None            # 部分 API 在最后一个 chunk 返回 usage


def parse_chat_response(raw: dict) -> LLMResponse:
    """解析非流式响应"""
    choice = raw["choices"][0]
    message = choice["message"]
    stop_reason = choice.get("finish_reason", "stop")
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    elif not isinstance(content, str):
        content = str(content)
    reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning_content, str):
        reasoning_content = str(reasoning_content)
    tool_calls = []

    for tc in (message.get("tool_calls") or []):
        func = tc["function"]
        tool_calls.append(ToolCall(
            id=tc["id"],
            name=func["name"],
            arguments=json.loads(func["arguments"]) if isinstance(func["arguments"], str) else func["arguments"],
        ))

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=raw.get("usage", {}),
        reasoning_content=reasoning_content,
    )


def parse_stream_line(line: str) -> StreamChunk | None:
    """解析 SSE 数据行"""
    if not line.startswith("data: "):
        return None
    data = line[6:]
    if data.strip() == "[DONE]":
        return StreamChunk(finish_reason="done")

    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None

    choice = obj.get("choices") or []
    if not choice:
        return None
    choice = choice[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")

    chunk = StreamChunk(
        content_delta=delta.get("content") or "",
        reasoning_delta=delta.get("reasoning_content") or "",
        finish_reason=finish,
    )

    # 提取 usage（部分 API 在最后一个 chunk 返回）
    if "usage" in obj and obj["usage"]:
        chunk.usage = obj["usage"]

    # 工具调用（增量）
    for tc in (delta.get("tool_calls") or []):
        chunk.tool_call_index = tc.get("index", 0)
        if "id" in tc:
            chunk.tool_call_id = tc["id"]
        func = tc.get("function", {})
        if "name" in func:
            chunk.tool_call_name = func["name"]
        if "arguments" in func:
            chunk.tool_call_arguments_delta = func["arguments"]

    return chunk


class LLMClient:
    """OpenAI 兼容 API 客户端（接 MiniMax）"""

    def __init__(self, api_key: str, base_url: str, model: str, proxy: str | None = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._closed = False
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            proxy=proxy,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        )

    async def __aenter__(self) -> LLMClient:
        if self._closed:
            raise RuntimeError("LLMClient is closed")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LLMClient is closed")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """非流式调用"""
        self._ensure_open()
        body = self._build_body(messages, tools, max_tokens, stream=False)
        resp = await self._http.post("/chat/completions", json=body)
        resp.raise_for_status()
        return parse_chat_response(resp.json())

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用，逐 chunk yield"""
        self._ensure_open()
        body = self._build_body(messages, tools, max_tokens, stream=True)
        logger.debug("LLM stream request: model=%s, messages=%d, tools=%d",
                      self.model, len(messages), len(tools or []))
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content_preview = str(m.get("content", ""))[:80]
            logger.debug("  [%d] %s: %s", i, role, content_preview)
        async with self._http.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error("LLM API error %d: %s", resp.status_code, error_body.decode(errors="replace"))
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    logger.debug("SSE line: %s", line[:200])
                chunk = parse_stream_line(line)
                if chunk is not None:
                    yield chunk

    async def collect_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        on_text: callable = None,
    ) -> LLMResponse:
        """流式调用，收集完整结果。on_text 回调用于实时输出文本。"""
        content_parts: list[str] = []
        tool_calls_build: dict[int, dict] = {}  # index → {id, name, arguments}
        finish_reason = "stop"
        usage_data: dict = {}

        async for chunk in self.chat_stream(messages, tools, max_tokens):
            # 推理过程（MiMo 等推理模型的思考过程）
            if chunk.reasoning_delta:
                if on_text:
                    on_text(chunk.reasoning_delta)

            # 文本增量
            if chunk.content_delta:
                content_parts.append(chunk.content_delta)
                if on_text:
                    on_text(chunk.content_delta)

            # 工具调用增量
            if chunk.tool_call_index is not None:
                idx = chunk.tool_call_index
                if idx not in tool_calls_build:
                    tool_calls_build[idx] = {"id": "", "name": "", "arguments": ""}
                tc = tool_calls_build[idx]
                if chunk.tool_call_id:
                    tc["id"] = chunk.tool_call_id
                if chunk.tool_call_name:
                    tc["name"] = chunk.tool_call_name
                if chunk.tool_call_arguments_delta:
                    tc["arguments"] += chunk.tool_call_arguments_delta

            if chunk.finish_reason and chunk.finish_reason != "done":
                finish_reason = chunk.finish_reason

            # 收集 usage（部分 API 在最后一个 chunk 返回）
            if chunk.usage:
                usage_data = chunk.usage

        # 组装 ToolCall 列表
        tool_calls = []
        for idx in sorted(tool_calls_build.keys()):
            tc = tool_calls_build[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=args))

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            usage=usage_data,
        )

    def _build_body(self, messages, tools, max_tokens, stream=False):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self._http.aclose()
