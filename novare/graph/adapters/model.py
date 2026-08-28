"""novare/graph/adapters/model.py — 模型适配层（ModelPort）

Graph 节点只依赖 ModelPort 协议，不直接依赖任何模型 SDK。

两个实现：
- LegacyModelPort    包装现有 LLMClient（httpx 直连 OpenAI 兼容 API），
                     完整保留 reasoning_content 等非标准字段。
- LangChainModelPort 包装 LangChain BaseChatModel（如 ChatOpenAI），
                     由 LangChain 负责流式解析 / tool_call 增量聚合。

NOVARE_MODEL_PORT=legacy|langchain 控制选择；auto 时默认 legacy
（MiniMax reasoning_content 兼容性未经全量验证前不冒险切换）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from novare.llm_client import LLMClient

logger = logging.getLogger("novare.graph.model")


# ── 统一数据结构 ──────────────────────────────────────────────


@dataclass
class ToolCallSpec:
    """模型请求的一次工具调用（与 LLMClient.ToolCall 等价，供 graph 层使用）"""

    id: str
    name: str
    arguments: dict


@dataclass
class ModelResult:
    """一次模型调用的聚合结果"""

    content: str
    tool_calls: list[ToolCallSpec]
    stop_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    reasoning_content: str = ""


@dataclass
class StreamDelta:
    """流式增量。text/reasoning 二选一。"""

    text: str = ""
    reasoning: str = ""


DeltaCallback = Callable[[StreamDelta], None]


@runtime_checkable
class ModelPort(Protocol):
    """模型调用端口协议。

    astream_collect: 流式调用并聚合完整结果；on_delta 用于实时推送文本增量。
    传输层重试由调用方（call_model 节点）通过 RetryExecutor 包裹，
    实现内部不做重试（与 LLMClient.collect_stream 现状一致）。
    """

    async def astream_collect(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> ModelResult: ...


# ── Legacy 实现：包装现有 LLMClient ───────────────────────────


class LegacyModelPort:
    """把现有 LLMClient 适配为 ModelPort。

    注意：LLMClient.collect_stream 会把 reasoning_delta 与 content_delta
    都写进 on_text —— 这里保持同样行为，确保 legacy / graph 两种运行时
    的流式输出对用户可见的内容一致。
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def astream_collect(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> ModelResult:
        def _on_text(chunk: str) -> None:
            if on_delta is not None:
                on_delta(StreamDelta(text=chunk))

        response = await self.client.collect_stream(messages, tools=tools, on_text=_on_text)
        return ModelResult(
            content=response.content,
            tool_calls=[
                ToolCallSpec(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in response.tool_calls
            ],
            stop_reason=response.stop_reason,
            usage=response.usage,
            reasoning_content=response.reasoning_content,
        )


# ── LangChain 实现：包装 BaseChatModel ────────────────────────


def _to_lc_message(m: dict):
    """OpenAI dict 消息 → LangChain 消息对象。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    role = m.get("role")
    content = m.get("content") or ""
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "tool":
        return ToolMessage(content=content, tool_call_id=m.get("tool_call_id", ""))
    # assistant：还原 tool_calls（args 必须是 dict，LangChain 自动序列化）
    tool_calls = []
    for tc in m.get("tool_calls") or []:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args
        tool_calls.append({
            "name": func.get("name", ""),
            "args": args,
            "id": tc.get("id", ""),
            "type": "tool_call",
        })
    return AIMessage(content=content, tool_calls=tool_calls)


def _usage_to_legacy_dict(usage_metadata: dict | None) -> dict:
    """LangChain usage_metadata → 现有 usage dict 键名（保持 usage_tracker 兼容）。"""
    if not usage_metadata:
        return {}
    result: dict = {}
    if "input_tokens" in usage_metadata:
        result["prompt_tokens"] = usage_metadata["input_tokens"]
    if "output_tokens" in usage_metadata:
        result["completion_tokens"] = usage_metadata["output_tokens"]
    if "total_tokens" in usage_metadata:
        result["total_tokens"] = usage_metadata["total_tokens"]
    return result


class LangChainModelPort:
    """把 LangChain BaseChatModel 适配为 ModelPort。

    流式解析、tool_call 增量聚合、usage 提取全部交给 LangChain；
    本类只负责 dict 消息转换和统一结果结构。
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def astream_collect(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> ModelResult:
        from langchain_core.messages import AIMessageChunk

        lc_messages = [_to_lc_message(m) for m in messages]
        model = self.chat_model.bind_tools(tools) if tools else self.chat_model

        accumulated: AIMessageChunk | None = None
        reasoning_parts: list[str] = []
        async for chunk in model.astream(lc_messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            # 文本增量（content 可能是 str 或分块 list，统一走 .text()）
            text = chunk.text()
            if text and on_delta is not None:
                on_delta(StreamDelta(text=text))
            # reasoning 增量：OpenAI 兼容服务放在 additional_kwargs.reasoning_content
            reasoning = chunk.additional_kwargs.get("reasoning_content") or ""
            if reasoning and on_delta is not None:
                on_delta(StreamDelta(reasoning=reasoning))
                reasoning_parts.append(reasoning)

        if accumulated is None:
            return ModelResult(content="", tool_calls=[], stop_reason="stop")

        content = accumulated.text()
        tool_calls: list[ToolCallSpec] = []
        for tc in accumulated.tool_calls or []:
            tool_calls.append(ToolCallSpec(
                id=tc.get("id") or "",
                name=tc.get("name", ""),
                arguments=tc.get("args") or {},
            ))
        # 参数解析失败的工具调用（invalid_tool_calls）也保留为空参数调用，
        # 避免静默丢失导致 tool result 协议不完整
        for bad in accumulated.invalid_tool_calls or []:
            tool_calls.append(ToolCallSpec(
                id=bad.get("id") or "",
                name=bad.get("name", ""),
                arguments={},
            ))

        finish = (accumulated.response_metadata or {}).get("finish_reason") or "stop"
        return ModelResult(
            content=content,
            tool_calls=tool_calls,
            stop_reason=finish,
            usage=_usage_to_legacy_dict(accumulated.usage_metadata),
            reasoning_content="".join(reasoning_parts),
        )


# ── 兼容适配：供 HybridContextCompactor 等需要 .chat() 的组件使用 ──


@dataclass
class _ChatCompatResponse:
    content: str
    usage: dict = field(default_factory=dict)


class ChatCompatLLM:
    """把 BaseChatModel 包装成 LLMClient.chat() 兼容的最小接口。

    用于 HybridContextCompactor（只调用 chat() 并读取 .content）。
    """

    def __init__(self, chat_model):
        self.chat_model = chat_model

    async def chat(self, messages: list[dict], tools=None, max_tokens: int = 4096):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages = [_to_lc_message(m) for m in messages]
        response: AIMessage = await self.chat_model.ainvoke(lc_messages)
        return _ChatCompatResponse(
            content=response.text(),
            usage=_usage_to_legacy_dict(response.usage_metadata),
        )


# ── 工厂 ──────────────────────────────────────────────────────


def build_langchain_chat_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 4096,
    timeout: float = 300.0,
):
    """构造 OpenAI 兼容的 ChatOpenAI 实例。

    streaming=True 与现有流式输出路径对齐；temperature 不设置，
    交给服务端默认（与 LLMClient._build_body 现状一致）。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        max_tokens=max_tokens,
        timeout=timeout,
        streaming=True,
    )


def build_model_port(
    *,
    port_name: str = "auto",
    llm_client: LLMClient,
    chat_model=None,
) -> ModelPort:
    """按配置选择 ModelPort 实现。

    - legacy / auto：LLMClient 包装（MiniMax reasoning_content 兼容）
    - langchain：使用传入的 chat_model（由上层负责构造，便于注入测试替身）
    """
    if port_name == "langchain":
        if chat_model is None:
            raise ValueError("NOVARE_MODEL_PORT=langchain 需要提供 chat_model")
        return LangChainModelPort(chat_model)
    return LegacyModelPort(llm_client)
