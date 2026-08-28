import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from novare.agent_loop import AgentLoop
from novare.config import NovareConfig
from novare.hallucination_verifier import (
    AtomicClaim,
    ClaimAssessment,
    ClaimImportance,
    EvidenceVerdict,
    HallucinationVerifier,
    VerificationResult,
    _aggregate_risk,
    _parse_json_object,
)
from novare.llm_client import LLMResponse, ToolCall
from novare.session import Session
from novare.subagents.types import SubagentType, get_allowlist
from novare.tools.registry import ToolDef, ToolRegistry


def _response(payload: dict) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(payload, ensure_ascii=False),
        tool_calls=[],
        stop_reason="stop",
        usage={},
    )


def _rag_result(text: str = "论文原文证据", *, ok: bool = True) -> str:
    if not ok:
        return json.dumps({
            "ok": False,
            "summary": "未找到相关内容",
            "error": "未找到相关内容",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "summary": "检索成功",
        "data": {
            "results": [{
                "paper_id": "paper-1",
                "chunk_id": "chunk-1",
                "title": "Test Paper",
                "section": "Results",
                "text": text,
                "rerank_score": 0.92,
            }],
        },
        "sources": [],
        "warnings": [],
    }, ensure_ascii=False)


def _make_registry(handler) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(ToolDef(
        name="rag_query",
        description="test rag",
        parameters={"type": "object", "properties": {}},
        handler=handler,
        source="mcp:test",
    ))
    return registry


@pytest.mark.asyncio
async def test_supported_claim_keeps_answer_and_user_scope():
    calls = []

    async def rag_handler(args, **kwargs):
        calls.append((args, kwargs))
        return _rag_result("实验准确率为 91%。")

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": [{
            "claim_id": "C1",
            "text": "实验准确率为 91%。",
            "importance": "high",
            "claim_type": "numeric",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "SUPPORTED",
            "evidence_ids": ["E1.1"],
            "reasoning": "原文直接支持。",
        }]}),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="实验准确率为 91%。",
        user_question="结果如何？",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "verified"
    assert result.corrected_answer == result.original_answer
    assert result.risk_score == 0
    assert result.rag_queries == 1
    assert len(calls) == 1
    assert calls[0][0]["question"] == "实验准确率为 91%。"
    assert calls[0][1]["user_id"] == "user-1"
    report = result.to_dict()
    assert report["evidence"] == [{
        "evidence_id": "E1.1",
        "claim_id": "C1",
        "paper_id": "paper-1",
        "chunk_id": "chunk-1",
        "title": "Test Paper",
        "section": "Results",
        "text": "实验准确率为 91%。",
        "score": 0.92,
    }]


@pytest.mark.asyncio
async def test_contradicted_claim_is_revised_and_high_risk():
    async def rag_handler(args, **kwargs):
        return _rag_result("论文报告准确率为 81%，不是 91%。")

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": [{
            "claim_id": "C1",
            "text": "论文报告准确率为 91%。",
            "importance": "high",
            "claim_type": "numeric",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "CONTRADICTED",
            "evidence_ids": ["E1.1"],
            "reasoning": "证据给出的数字不同。",
        }]}),
        _response({
            "corrected_answer": "论文报告准确率为 81%。",
            "changes": ["修正准确率"],
        }),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="论文报告准确率为 91%。",
        user_question="准确率是多少？",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "revised"
    assert result.corrected_answer == "论文报告准确率为 81%。"
    assert result.did_revise is True
    assert result.risk_level == "high"
    assert result.max_claim_risk == 1.0
    assert result.llm_calls == 3


@pytest.mark.asyncio
async def test_missing_evidence_is_nei_not_contradicted():
    async def rag_handler(args, **kwargs):
        return _rag_result(ok=False)

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": [{
            "claim_id": "C1",
            "text": "该方法在所有数据集上最好。",
            "importance": "medium",
            "claim_type": "comparison",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "evidence_ids": [],
            "reasoning": "没有检索到足够证据。",
        }]}),
        _response({
            "corrected_answer": "现有证据不足以确认该方法在所有数据集上最好。",
            "changes": ["弱化无证据比较"],
        }),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="该方法在所有数据集上最好。",
        user_question="这个方法最好吗？",
        tool_context={"user_id": "user-1"},
    )

    assert result.assessments[0].verdict == EvidenceVerdict.NOT_ENOUGH_EVIDENCE
    assert result.risk_level == "medium"
    assert result.warnings


@pytest.mark.asyncio
async def test_repair_failure_returns_annotated_draft():
    async def rag_handler(args, **kwargs):
        return _rag_result(ok=False)

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": [{
            "claim_id": "C1",
            "text": "未经证实的结论。",
            "importance": "high",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "evidence_ids": [],
            "reasoning": "证据不足。",
        }]}),
        _response({"corrected_answer": "", "changes": []}),
        _response({"wrong": "shape"}),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="未经证实的结论。",
        user_question="结论是什么？",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "repair_failed"
    assert "证据核验提示" in result.corrected_answer
    assert "证据不足" in result.corrected_answer


@pytest.mark.asyncio
async def test_invalid_claim_schema_retries_once():
    async def rag_handler(args, **kwargs):
        return _rag_result()

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": "invalid"}),
        _response({"claims": []}),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="纯观点，没有事实。",
        user_question="你怎么看？",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "no_verifiable_claims"
    assert result.llm_calls == 2
    assert llm.chat.await_count == 2


@pytest.mark.asyncio
async def test_supported_without_evidence_is_rejected_and_retried():
    async def rag_handler(args, **kwargs):
        return _rag_result("直接支持该事实。")

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=[
        _response({"claims": [{
            "claim_id": "C1",
            "text": "可验证事实。",
            "importance": "medium",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "SUPPORTED",
            "evidence_ids": [],
            "reasoning": "未引用证据。",
        }]}),
        _response({"assessments": [{
            "claim_id": "C1",
            "verdict": "SUPPORTED",
            "evidence_ids": ["E1.1"],
            "reasoning": "证据直接支持。",
        }]}),
    ])
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )

    result = await verifier.verify(
        answer="可验证事实。",
        user_question="请说明事实。",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "verified"
    assert result.llm_calls == 3
    assert llm.chat.await_count == 3


@pytest.mark.asyncio
async def test_verifier_timeout_returns_original_answer():
    async def rag_handler(args, **kwargs):
        return _rag_result()

    blocker = asyncio.Event()

    async def slow_chat(*args, **kwargs):
        await blocker.wait()

    llm = AsyncMock()
    llm.chat = slow_chat
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(rag_handler),
        enabled=True,
    )
    verifier.timeout = 0.01

    result = await verifier.verify(
        answer="原始回答",
        user_question="问题",
        tool_context={"user_id": "user-1"},
    )

    assert result.status == "timeout"
    assert result.corrected_answer == "原始回答"
    assert result.risk_level == "unknown"


@pytest.mark.asyncio
async def test_empty_final_model_content_returns_actionable_failure():
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[],
        stop_reason="length",
        usage={},
        reasoning_content="analysis" * 100,
    ))
    verifier = HallucinationVerifier(
        llm_client=llm,
        tool_executor=_make_registry(AsyncMock()),
        enabled=True,
    )

    result = await verifier.verify(answer="需要核验的回答。", user_question="问题是什么？")

    assert result.status == "failed"
    assert result.risk_level == "unknown"
    assert any("no final JSON content" in warning for warning in result.warnings)


def test_risk_aggregation_preserves_max_claim_risk():
    claims = [
        AtomicClaim("C1", "关键数字错误", ClaimImportance.HIGH),
        AtomicClaim("C2", "低风险事实", ClaimImportance.LOW),
    ]
    assessments = [
        ClaimAssessment("C1", EvidenceVerdict.CONTRADICTED, ("E1",)),
        ClaimAssessment("C2", EvidenceVerdict.SUPPORTED, ("E2",)),
    ]

    weighted, score, max_risk, level = _aggregate_risk(claims, assessments)

    assert weighted[0].risk == 1.0
    assert 0 < score < 1
    assert max_risk == 1.0
    assert level == "high"


def test_verifier_subagent_allowlist_is_read_only():
    allowed = get_allowlist(SubagentType.VERIFIER)
    assert "rag_query" in allowed
    assert "read_file" in allowed
    assert "write_file" not in allowed
    assert "edit_file" not in allowed
    assert "code_execute" not in allowed


def test_json_parser_rejects_multiple_objects():
    with pytest.raises(ValueError):
        _parse_json_object('{"claims": []}{"claims": []}')


def test_verifier_config_loads_and_clamps(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVARE_HALLUCINATION_VERIFIER_ENABLED", "true")
    monkeypatch.setenv("NOVARE_HALLUCINATION_VERIFIER_MAX_CLAIMS", "99")
    monkeypatch.setenv("NOVARE_HALLUCINATION_VERIFIER_TOP_K", "0")
    monkeypatch.setenv("NOVARE_HALLUCINATION_VERIFIER_CONCURRENCY", "99")
    monkeypatch.setenv("NOVARE_HALLUCINATION_VERIFIER_TIMEOUT", "1")

    config = NovareConfig.load(tmp_path / "missing.json")

    assert config.hallucination_verifier_enabled is True
    assert config.hallucination_verifier_max_claims == 15
    assert config.hallucination_verifier_top_k == 1
    assert config.hallucination_verifier_concurrency == 8
    assert config.hallucination_verifier_timeout == 10


@pytest.mark.asyncio
async def test_agent_loop_buffers_rag_draft_and_returns_verified_answer():
    async def rag_handler(args, **kwargs):
        return _rag_result()

    registry = _make_registry(rag_handler)
    llm = AsyncMock()
    llm.collect_stream = AsyncMock(side_effect=[
        LLMResponse(
            content="",
            tool_calls=[ToolCall("call-1", "rag_query", {"question": "q"})],
            stop_reason="tool_calls",
            usage={},
        ),
        LLMResponse(
            content="未经验证的草稿",
            tool_calls=[],
            stop_reason="stop",
            usage={},
        ),
    ])
    verifier = AsyncMock()
    verifier.enabled = True
    verifier.verify = AsyncMock(return_value=VerificationResult(
        original_answer="未经验证的草稿",
        corrected_answer="已验证的回答",
        status="revised",
        did_revise=True,
    ))
    loop = AgentLoop(
        llm_client=llm,
        tool_registry=registry,
        hallucination_verifier=verifier,
    )
    chunks = []
    reports = []

    result = await loop.run_turn(
        Session(),
        "请基于论文回答",
        on_text=chunks.append,
        on_verification=reports.append,
        tool_context={"user_id": "user-1"},
    )

    assert result == "已验证的回答"
    verifier.verify.assert_awaited_once()
    assert llm.collect_stream.await_args_list[1].kwargs["on_text"] is None
    assert chunks == ["已验证的回答"]
    assert reports[0]["status"] == "revised"


@pytest.mark.asyncio
async def test_agent_loop_does_not_verify_without_successful_rag():
    registry = ToolRegistry()
    llm = AsyncMock()
    llm.collect_stream = AsyncMock(return_value=LLMResponse(
        content="普通回答",
        tool_calls=[],
        stop_reason="stop",
        usage={},
    ))
    verifier = AsyncMock()
    verifier.enabled = True
    loop = AgentLoop(
        llm_client=llm,
        tool_registry=registry,
        hallucination_verifier=verifier,
    )

    result = await loop.run_turn(Session(), "你好")

    assert result == "普通回答"
    verifier.verify.assert_not_awaited()
