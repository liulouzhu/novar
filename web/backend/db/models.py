import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, BigInteger, Text, Boolean, DateTime, Float, LargeBinary,
    ForeignKey, UniqueConstraint, Index, CheckConstraint, JSON,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.orm import relationship

from .base import Base


# JSON 类型：PostgreSQL 使用 JSONB，SQLite/其他使用 JSON
JSON_TYPE = JSON().with_variant(PG_JSONB(), "postgresql")


def utcnow():
    return datetime.now(timezone.utc)


def gen_uuid():
    return uuid.uuid4()


class GUID(TypeDecorator):
    """UUID 类型适配器：PostgreSQL 用原生 UUID，SQLite 用 CHAR(36)。"""
    impl = String
    cache_ok = True

    def __init__(self):
        super().__init__(length=36)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return parsed  # asyncpg 接受 uuid.UUID
        return str(parsed)  # SQLite 需要字符串

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import UUID as PG_UUID
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))


class User(Base):
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class Paper(Base):
    """Shared paper metadata — public papers visible to all, private papers owner-only."""
    __tablename__ = "papers"
    __table_args__ = (
        CheckConstraint("visibility IN ('public', 'private')", name="ck_paper_visibility"),
    )

    id = Column(String(255), primary_key=True)  # doi:xxx, arxiv:xxx, s2:xxx
    title = Column(Text, nullable=False)
    authors = Column(JSON_TYPE, default=list)
    abstract = Column(Text)
    year = Column(Integer)
    source = Column(String(50))
    pdf_path = Column(Text)
    url = Column(Text)
    citation_count = Column(Integer, default=0)
    visibility = Column(String(10), nullable=False, default="public", server_default="public")
    created_by_user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class PaperIdentifier(Base):
    """Canonical external identifiers that resolve to one shared paper."""
    __tablename__ = "paper_identifiers"
    __table_args__ = (
        UniqueConstraint("identifier", name="uq_paper_identifier"),
        Index("idx_paper_identifiers_paper", "paper_id"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    paper_id = Column(String(255), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    identifier_type = Column(String(20), nullable=False)
    identifier = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    paper = relationship("Paper")


class FileBlob(Base):
    """Content-addressed file stored once globally and authorized through UserUpload."""
    __tablename__ = "file_blobs"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_file_blob_sha256"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    sha256 = Column(String(64), nullable=False, index=True)
    size_bytes = Column(BigInteger, nullable=False)
    mime_type = Column(String(255))
    storage_path = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class UserUpload(Base):
    """User-scoped authorization and metadata for a globally deduplicated blob."""
    __tablename__ = "user_uploads"
    __table_args__ = (
        UniqueConstraint("user_id", "blob_id", name="uq_user_upload_blob"),
        Index("idx_user_uploads_user", "user_id"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    blob_id = Column(GUID(), ForeignKey("file_blobs.id"), nullable=False, index=True)
    original_filename = Column(String(512), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User")
    blob = relationship("FileBlob")


class PaperFile(Base):
    """A paper version backed by a deduplicated file blob."""
    __tablename__ = "paper_files"
    __table_args__ = (
        UniqueConstraint("paper_id", "blob_id", name="uq_paper_file_blob"),
        CheckConstraint("access_scope IN ('public', 'private')", name="ck_paper_file_access_scope"),
        Index("idx_paper_files_blob", "blob_id"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    paper_id = Column(String(255), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    blob_id = Column(GUID(), ForeignKey("file_blobs.id"), nullable=False)
    source = Column(String(30), nullable=False, default="upload", server_default="upload")
    version = Column(String(50))
    access_scope = Column(String(10), nullable=False, default="private", server_default="private")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    paper = relationship("Paper")
    blob = relationship("FileBlob")


class UserPaper(Base):
    """User-paper relationship with typed access levels."""
    __tablename__ = "user_papers"
    __table_args__ = (
        UniqueConstraint("user_id", "paper_id", name="uq_user_paper"),
        CheckConstraint(
            "relation_type IN ('searched', 'parsed', 'uploaded', 'shared')",
            name="ck_user_paper_relation_type",
        ),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    paper_id = Column(String(255), ForeignKey("papers.id"), nullable=False, index=True)
    relation_type = Column(String(20), nullable=False, default="searched", server_default="searched")
    has_fulltext_access = Column(Boolean, nullable=False, default=False, server_default="false")
    source = Column(String(30))  # paper_search | paper_parse | upload | share
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    parsed_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    user = relationship("User")
    paper = relationship("Paper")


class Chunk(Base):
    """Shared paper chunks — generated once per paper."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(255), ForeignKey("papers.id"), nullable=False, index=True)
    section = Column(String(255))
    ordinal = Column(Integer)
    text = Column(Text, nullable=False)


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (
        Index("idx_citations_source", "source_id"),
        Index("idx_citations_target", "target_id"),
    )

    source_id = Column(String(255), ForeignKey("papers.id"), primary_key=True)
    target_id = Column(String(255), ForeignKey("papers.id"), primary_key=True)


class Embedding(Base):
    """Vector embedding for a chunk — stored as numpy float32 bytes."""
    __tablename__ = "embeddings"

    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    dim = Column(Integer, nullable=False)
    vec = Column(LargeBinary, nullable=False)  # numpy.float32 tobytes()


class PaperCleanupJob(Base):
    """Durable outbox job for idempotent cross-store paper cleanup."""
    __tablename__ = "paper_cleanup_jobs"
    __table_args__ = (
        CheckConstraint("scope IN ('user', 'paper')", name="ck_paper_cleanup_scope"),
        CheckConstraint(
            "status IN ('pending', 'running', 'failed', 'completed')",
            name="ck_paper_cleanup_status",
        ),
        Index("idx_paper_cleanup_ready", "status", "next_retry_at"),
        Index("idx_paper_cleanup_paper", "paper_id"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    paper_id = Column(String(255), nullable=False)
    user_id = Column(GUID(), nullable=False)
    scope = Column(String(10), nullable=False)
    status = Column(String(20), nullable=False, default="pending", server_default="pending")
    steps = Column(JSON_TYPE, nullable=False, default=dict)
    payload = Column(JSON_TYPE, nullable=False, default=dict)
    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    last_error = Column(Text)
    next_retry_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class SessionModel(Base):
    """Chat session — user-scoped."""
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)  # keep existing format: timestamp-uuid
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(Text, default="New Chat")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    # 批量记忆提取游标
    last_extracted_message_id = Column(Integer, nullable=True)
    last_memory_extracted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User")


class MessageModel(Base):
    """Chat message — scoped via session."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, tool, system
    content = Column(Text)
    tool_calls = Column(JSON_TYPE)  # assistant tool_calls array
    tool_call_id = Column(String(128))  # for tool role messages
    name = Column(String(128))  # tool name
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class ContextSnapshot(Base):
    """Compacted working context while raw messages remain immutable."""
    __tablename__ = "context_snapshots"

    session_id = Column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    snapshot_data = Column(JSON_TYPE, nullable=False)
    compacted_through_message_id = Column(Integer, nullable=False)
    schema_version = Column(Integer, nullable=False, default=1, server_default="1")
    estimated_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    session = relationship("SessionModel")
    user = relationship("User")


class RecoveryStateModel(Base):
    """PR 2：执行恢复状态 — 追踪 tool_call 协议完整性。

    每次 run_turn 创建一个 RecoveryState，记录 tool_call_id 的执行状态、
    幂等性 key 和动作指纹，支持进程中断后确定性对账。
    """
    __tablename__ = "recovery_states"
    __table_args__ = (
        UniqueConstraint("session_id", "run_id", name="uq_recovery_state_session_run"),
        Index("idx_recovery_states_session", "session_id"),
        Index("idx_recovery_states_run", "run_id"),
        Index("idx_recovery_states_user", "user_id"),
        CheckConstraint(
            "run_status IN ('running', 'completed', 'failed', 'cancelled', 'timed_out', 'interrupted', 'recovered')",
            name="ck_recovery_state_run_status",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    run_id = Column(String(32), nullable=False)
    turn_id = Column(String(32), nullable=False)
    run_status = Column(String(20), nullable=False, default="running", server_default="running")
    iteration = Column(Integer, nullable=False, default=0)
    retry_count = Column(Integer, nullable=False, default=0)
    recovery_data = Column(JSON_TYPE, nullable=False, default=dict)
    schema_version = Column(Integer, nullable=False, default=2, server_default="2")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    session = relationship("SessionModel")
    user = relationship("User")


class RecoveryEventModel(Base):
    """PR 2：恢复事件日志 — 支持中断恢复和对账。

    每个工具调用的生命周期事件都被持久化，支持：
    - 进程中断后恢复
    - 幂等性检查
    - 审计追踪
    """
    __tablename__ = "recovery_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_recovery_event_run_sequence"),
        UniqueConstraint("run_id", "event_key", name="uq_recovery_event_run_key"),
        Index("idx_recovery_events_run", "run_id"),
        Index("idx_recovery_events_session", "session_id"),
        Index("idx_recovery_events_user", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    run_id = Column(String(32), nullable=False)
    sequence = Column(Integer, nullable=False)
    event_key = Column(String(128), nullable=False)  # e.g., "tc1:TOOL_COMPLETED"
    event_type = Column(String(50), nullable=False)  # e.g., "TOOL_COMPLETED"
    payload = Column(JSON_TYPE, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    session = relationship("SessionModel")
    user = relationship("User")


class KnowledgeNode(Base):
    __tablename__ = "knowledge_nodes"

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    label = Column(String(255), nullable=False)
    type = Column(String(50))  # concept, paper, author, method, dataset, task
    properties = Column(JSON_TYPE, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_edges"
    __table_args__ = (
        UniqueConstraint("user_id", "source_node_id", "target_node_id", "relation_type", name="uq_user_edge"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    source_node_id = Column(GUID(), ForeignKey("knowledge_nodes.id"), nullable=False)
    target_node_id = Column(GUID(), ForeignKey("knowledge_nodes.id"), nullable=False)
    relation_type = Column(String(100), nullable=False)
    properties = Column(JSON_TYPE, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class ChannelUser(Base):
    """渠道用户映射 — 将平台 sender_id 关联到 Novare user_id。"""
    __tablename__ = "channel_users"
    __table_args__ = (
        UniqueConstraint("channel", "platform_user_id", name="uq_channel_user"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    novare_user_id = Column(GUID(), ForeignKey("users.id"), nullable=False, index=True)
    channel = Column(String(32), nullable=False)           # weixin, telegram, ...
    platform_user_id = Column(String(128), nullable=False)  # 平台用户 ID
    platform_username = Column(String(128))                 # 平台昵称（可选）
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User")


class UserMemory(Base):
    """用户长期记忆 — 自动从对话中提取的用户偏好。"""
    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "category", "key", name="uq_user_memory_key"),
        Index("idx_user_memories_user", "user_id"),
        Index("idx_user_memories_category", "user_id", "category"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    category = Column(String(50), nullable=False)    # "research_preference" | "interaction_preference"
    key = Column(String(100), nullable=False)         # 如 "research_field", "preferred_language"
    value = Column(Text, nullable=False)              # 具体值
    confidence = Column(Float, default=1.0)           # 置信度 0-1
    pinned = Column(Boolean, default=False)           # 锁定：pinned 的记忆不参与淘汰
    tags = Column(JSON_TYPE, default=list)                # 标签列表
    source = Column(String(50), default="auto")       # "auto" | "user"
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User")


class EpisodicMemory(Base):
    """情景记忆 — 用户完成过的研究任务、决策、实验结果等。"""
    __tablename__ = "episodic_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "content_hash", name="uq_episodic_memory_hash"),
        Index("idx_episodic_memories_user", "user_id"),
        Index("idx_episodic_memories_session", "session_id"),
        Index("idx_episodic_memories_status", "user_id", "status"),
    )

    id = Column(GUID(), primary_key=True, default=gen_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    session_id = Column(String(64), nullable=True)

    memory_type = Column(String(50), nullable=False)
    summary = Column(String(500), nullable=False)
    context = Column(Text, default="")
    action = Column(Text, default="")
    outcome = Column(Text, default="")

    topics = Column(JSON_TYPE, default=list)
    source_message_ids = Column(JSON_TYPE, default=list)

    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)

    content_hash = Column(String(64), nullable=False)
    embedding_model = Column(String(100), default="")
    vector_id = Column(String(64), nullable=True)
    index_status = Column(String(20), nullable=False, default="pending")

    status = Column(String(20), nullable=False, default="active")
    pinned = Column(Boolean, default=False)

    occurred_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    last_retrieved_at = Column(DateTime(timezone=True), nullable=True)
    retrieval_count = Column(Integer, default=0)

    user = relationship("User")
