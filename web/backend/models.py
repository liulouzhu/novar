"""Pydantic 请求/响应模型"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── WebSocket 事件 ──────────────────────────────────────────

class WSEvent(BaseModel):
    """服务端 → 客户端 的 WebSocket 事件"""
    type: str
    # 通用可选字段，按 type 取用
    content: str | None = None
    tool: str | None = None
    tool_id: str | None = None
    params: dict | None = None
    result: str | None = None
    duration: float | None = None
    error: str | None = None
    stage: str | None = None
    message: str | None = None
    usage: dict | None = None


class WSIncoming(BaseModel):
    """客户端 → 服务端 的 WebSocket 消息"""
    type: str  # "send" | "send_with_refs"
    content: str = ""
    references: list[dict] | None = None
    skill: str | None = None


# ── REST 请求/响应 ──────────────────────────────────────────

class SessionMeta(BaseModel):
    session_id: str
    title: str = ""
    message_count: int = 0
    updated_at: str = ""


class SessionDetail(BaseModel):
    session_id: str
    messages: list[dict]
    title: str = ""


class PaperOut(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    source: str | None = None
    url: str | None = None
    pdf_path: str | None = None
    citation_count: int = 0
    is_parsed: bool = False
    created_at: str | None = None


class PaperFullTextSection(BaseModel):
    section: str
    text: str
    chunk_count: int = 0


class PaperFullTextOut(BaseModel):
    paper_id: str
    title: str
    sections: list[PaperFullTextSection] = Field(default_factory=list)
    content: str = ""


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    already_uploaded: bool = False
    file_path: str | None = None
    message: str


# ── Memory ─────────────────────────────────────────────────

class MemoryOut(BaseModel):
    id: int
    category: str
    key: str
    value: str
    confidence: float
    pinned: bool
    tags: list[str] = Field(default_factory=list)
    source: str = "auto"
    created_at: str | None = None
    updated_at: str | None = None


class MemoryUpdate(BaseModel):
    value: str | None = None
    tags: list[str] | None = None
    confidence: float | None = None
