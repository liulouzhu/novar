"""Tests for the optional Qwen3 rerank stage."""

from unittest.mock import AsyncMock, patch

import pytest


class _FakeResponse:
    def __init__(self, payload: dict | None = None, text: str = ""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, captured: dict,
                 get_response: _FakeResponse | None = None, **kwargs):
        self.response = response
        self.get_response = get_response
        self.captured = captured
        self.captured["client_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.captured["url"] = url
        self.captured.update(kwargs)
        return self.response

    async def get(self, url, **kwargs):
        self.captured["get_url"] = url
        self.captured["get_kwargs"] = kwargs
        return self.get_response


@pytest.mark.asyncio
async def test_qwen3_rerank_reorders_candidates(monkeypatch):
    from core import reranker

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("RAG_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANK_URL", raising=False)
    monkeypatch.delenv("RAG_RERANK_PROVIDER", raising=False)
    captured = {}
    response = _FakeResponse({
        "output": {
            "results": [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.31},
            ]
        }
    })

    def client_factory(**kwargs):
        return _FakeAsyncClient(response, captured, **kwargs)

    candidates = [
        {"chunk_id": 1, "title": "A", "section": "S", "text": "first"},
        {"chunk_id": 2, "title": "B", "section": "S", "text": "second"},
    ]
    monkeypatch.setattr(reranker.httpx, "AsyncClient", client_factory)

    result = await reranker.rerank_chunks("query", candidates)

    assert result.available is True
    assert [hit["chunk_id"] for hit in result.hits] == [2, 1]
    assert result.hits[0]["rerank_score"] == 0.98
    assert captured["json"]["model"] == "qwen3-rerank"
    assert captured["json"]["parameters"]["top_n"] == 2
    assert captured["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_gradio_rerank_reorders_candidates_without_api_key(monkeypatch):
    from core import reranker

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANK_API_KEY", raising=False)
    monkeypatch.setenv("RAG_RERANK_URL", "http://192.168.137.242:7861")
    monkeypatch.setenv("RAG_RERANK_PROVIDER", "gradio")
    captured = {}
    job_response = _FakeResponse({"event_id": "event-123"})
    stream_response = _FakeResponse(text=(
        "event: complete\n"
        'data: [[{"text":"second","score":0.98,"rank":1},'
        '{"text":"first","score":0.31,"rank":2}]]\n\n'
    ))

    def client_factory(**kwargs):
        return _FakeAsyncClient(job_response, captured, stream_response, **kwargs)

    candidates = [
        {"chunk_id": 1, "text": "first"},
        {"chunk_id": 2, "text": "second"},
    ]
    monkeypatch.setattr(reranker.httpx, "AsyncClient", client_factory)

    result = await reranker.rerank_chunks("query", candidates)

    assert result.available is True
    assert [hit["chunk_id"] for hit in result.hits] == [2, 1]
    assert result.hits[0]["rerank_score"] == 0.98
    assert captured["url"] == "http://192.168.137.242:7861/gradio_api/call/v2/do_rerank"
    assert captured["get_url"].endswith("/gradio_api/call/do_rerank/event-123")
    assert captured["json"] == {
        "query": "query", "passages_text": "first\nsecond", "top_k": 2,
    }
    assert "headers" not in captured


@pytest.mark.asyncio
async def test_rerank_missing_key_is_safe(monkeypatch):
    from core.reranker import rerank_chunks

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("RAG_RERANK_URL", raising=False)
    monkeypatch.setenv("RAG_RERANK_PROVIDER", "dashscope")
    result = await rerank_chunks("query", [{"chunk_id": 1, "text": "doc"}])

    assert result.available is False
    assert result.hits == []
    assert "unset" in result.error


@pytest.mark.asyncio
async def test_rag_query_uses_reranked_order():
    import tools.rag_query as rq

    vector_hits = [
        {"score": 0.9, "chunk_id": 1, "text": "first", "paper_id": "p1",
         "title": "T", "section": "s", "source": "vector"},
        {"score": 0.8, "chunk_id": 2, "text": "second", "paper_id": "p1",
         "title": "T", "section": "s", "source": "vector"},
    ]

    async def fake_rerank(question, candidates):
        reranked = [dict(candidates[1]), dict(candidates[0])]
        for rank, hit in enumerate(reranked, 1):
            hit["rerank_rank"] = rank
            hit["rerank_score"] = 1.0 / rank
        return reranked, True, None

    with patch.object(rq, "RERANK_ENABLED", True), \
         patch.object(rq, "_get_user_paper_ids", return_value={"p1"}), \
         patch.object(rq, "embed_text_async", return_value=[0.1] * 1024), \
         patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=vector_hits), \
         patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)), \
         patch.object(rq, "_rerank_results", side_effect=fake_rerank):
        import json
        result = json.loads(await rq.handle_rag_query(
            {"question": "test", "top_k": 2}, user_id="u-1",
        ))

    assert result["ok"] is True
    assert [hit["chunk_id"] for hit in result["data"]["results"]] == [2, 1]
    assert result["data"]["rerank_applied"] is True
    assert "qwen3-rerank" in result["data"]["search_method"]


@pytest.mark.asyncio
async def test_rag_query_falls_back_to_rrf_when_rerank_fails():
    import json
    import tools.rag_query as rq

    vector_hits = [
        {"score": 0.9, "chunk_id": 1, "text": "first", "paper_id": "p1",
         "title": "T", "section": "s", "source": "vector"},
    ]
    with patch.object(rq, "RERANK_ENABLED", True), \
         patch.object(rq, "_get_user_paper_ids", return_value={"p1"}), \
         patch.object(rq, "embed_text_async", return_value=[0.1] * 1024), \
         patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=vector_hits), \
         patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)), \
         patch.object(rq, "_rerank_results", new_callable=AsyncMock,
                      return_value=([], False, "timeout")):
        result = json.loads(await rq.handle_rag_query(
            {"question": "test"}, user_id="u-1",
        ))

    assert result["ok"] is True
    assert result["data"]["rerank_applied"] is False
    assert result["data"]["results"][0]["chunk_id"] == 1
    assert any("kept RRF ordering" in warning for warning in result["warnings"])
