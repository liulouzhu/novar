"""novare/graph/adapters — LangChain 适配层

model.py 提供 ModelPort 协议与两种实现（Legacy / LangChain）。
"""

from novare.graph.adapters.model import (
    ChatCompatLLM,
    LangChainModelPort,
    LegacyModelPort,
    ModelPort,
    ModelResult,
    StreamDelta,
    ToolCallSpec,
    build_langchain_chat_model,
    build_model_port,
)

__all__ = [
    "ChatCompatLLM",
    "LangChainModelPort",
    "LegacyModelPort",
    "ModelPort",
    "ModelResult",
    "StreamDelta",
    "ToolCallSpec",
    "build_langchain_chat_model",
    "build_model_port",
]
