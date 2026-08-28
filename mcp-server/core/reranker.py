"""Reranker client used after RRF candidate fusion.

Environment variables:
    RAG_RERANK_PROVIDER     ``dashscope``, ``gradio``, or ``auto`` (default).
    RAG_RERANK_API_KEY      Optional DashScope key; falls back to DASHSCOPE_API_KEY.
    RAG_RERANK_URL          DashScope endpoint or Gradio service base URL.
    RAG_RERANK_MODEL        DashScope model name, default qwen3-rerank.
    RAG_RERANK_TIMEOUT      Request timeout in seconds.
    RAG_RERANK_MAX_DOC_CHARS Maximum characters sent for each candidate.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "rerank/text-rerank/text-rerank"
)


@dataclass(frozen=True)
class RerankResult:
    """Result that keeps provider failures separate from valid rankings."""

    hits: list[dict]
    available: bool
    error: str | None = None


def _candidate_document(candidate: dict, max_chars: int) -> str:
    parts = []
    title = str(candidate.get("title") or "").strip()
    section = str(candidate.get("section") or "").strip()
    text = str(candidate.get("text") or "").strip()
    if title:
        parts.append(f"Title: {title}")
    if section:
        parts.append(f"Section: {section}")
    if text:
        parts.append(text)
    return "\n".join(parts)[:max_chars]


def _rerank_provider(url: str) -> str:
    """Resolve the provider, keeping the established DashScope default."""
    configured = os.getenv("RAG_RERANK_PROVIDER", "auto").strip().lower()
    if configured in {"dashscope", "gradio"}:
        return configured
    if configured != "auto":
        raise ValueError("RAG_RERANK_PROVIDER must be dashscope, gradio, or auto")
    return "dashscope" if url == DEFAULT_RERANK_URL else "gradio"


def _gradio_endpoint(url: str) -> str:
    """Accept either a Gradio base URL or its do_rerank API endpoint."""
    normalized = url.rstrip("/")
    if normalized.endswith("/gradio_api/call/v2/do_rerank"):
        return normalized
    return f"{normalized}/gradio_api/call/v2/do_rerank"


def _gradio_complete_payload(stream: str) -> Any:
    """Extract the JSON payload attached to Gradio's SSE ``complete`` event."""
    event = ""
    data_lines: list[str] = []
    for line in stream.splitlines():
        if not line:
            if event == "complete" and data_lines:
                return json.loads("\n".join(data_lines))
            event = ""
            data_lines = []
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if event == "complete" and data_lines:
        return json.loads("\n".join(data_lines))
    raise ValueError("Gradio response did not contain a complete event")


def _rank_gradio_results(
    payload: Any, candidates: list[dict], documents: list[str],
) -> RerankResult:
    """Map Gradio's returned texts and scores back to their source candidates."""
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        return RerankResult([], False, "invalid Gradio rerank response")

    indexes_by_document: dict[str, list[int]] = {}
    for index, document in enumerate(documents):
        indexes_by_document.setdefault(document, []).append(index)

    ranked: list[dict] = []
    try:
        for item in payload[0]:
            document = item["text"]
            indexes = indexes_by_document.get(document)
            if not indexes:
                return RerankResult([], False, "Gradio result references an unknown document")
            hit = dict(candidates[indexes.pop(0)])
            hit["rerank_score"] = float(item["score"])
            hit["rerank_rank"] = len(ranked) + 1
            ranked.append(hit)
    except (KeyError, TypeError, ValueError) as exc:
        return RerankResult([], False, f"invalid Gradio rerank result item: {exc}")

    if len(ranked) != len(candidates):
        return RerankResult(
            [], False,
            f"Gradio rerank returned {len(ranked)} results for {len(candidates)} candidates",
        )
    return RerankResult(ranked, True)


async def _rerank_with_gradio(
    client: httpx.AsyncClient, url: str, query: str, documents: list[str],
    candidates: list[dict],
) -> RerankResult:
    endpoint = _gradio_endpoint(url)
    response = await client.post(
        endpoint,
        json={
            "query": query,
            "passages_text": "\n".join(documents),
            "top_k": len(documents),
        },
    )
    response.raise_for_status()
    event_id = response.json().get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return RerankResult([], False, "Gradio rerank response is missing event_id")

    stream_url = endpoint.replace("/v2/do_rerank", f"/do_rerank/{event_id}")
    stream_response = await client.get(stream_url)
    stream_response.raise_for_status()
    return _rank_gradio_results(
        _gradio_complete_payload(stream_response.text), candidates, documents,
    )


async def rerank_chunks(query: str, candidates: list[dict]) -> RerankResult:
    """Rerank RRF candidates with a DashScope or Gradio provider.

    The returned dictionaries are copies of the input candidates with
    ``rerank_score`` and ``rerank_rank`` added. Provider/configuration errors
    are returned as ``available=False`` so callers can safely retain RRF order.
    """
    if not candidates:
        return RerankResult([], True)

    url = os.getenv("RAG_RERANK_URL", DEFAULT_RERANK_URL).strip()
    model = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank").strip()
    try:
        timeout = max(1.0, float(os.getenv("RAG_RERANK_TIMEOUT", "30")))
        max_chars = max(256, int(os.getenv("RAG_RERANK_MAX_DOC_CHARS", "6000")))
        provider = _rerank_provider(url)
    except ValueError as exc:
        return RerankResult([], False, f"invalid rerank configuration: {exc}")

    documents = [_candidate_document(item, max_chars) for item in candidates]
    if provider == "gradio":
        try:
            # Private Gradio endpoints must not be routed through HTTP_PROXY.
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                return await _rerank_with_gradio(
                    client, url, query, documents, candidates,
                )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return RerankResult([], False, str(exc))

    api_key = os.getenv("RAG_RERANK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return RerankResult(
            [], False,
            "RAG_RERANK_API_KEY and DASHSCOPE_API_KEY are both unset",
        )

    body = {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": {
            "return_documents": False,
            "top_n": len(documents),
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return RerankResult([], False, str(exc))

    output = payload.get("output", payload)
    raw_results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(raw_results, list):
        return RerankResult([], False, "rerank response is missing output.results")

    ranked: list[dict] = []
    seen: set[int] = set()
    try:
        for item in raw_results:
            index = int(item["index"])
            if index < 0 or index >= len(candidates) or index in seen:
                continue
            hit = dict(candidates[index])
            hit["rerank_score"] = float(item["relevance_score"])
            hit["rerank_rank"] = len(ranked) + 1
            ranked.append(hit)
            seen.add(index)
    except (KeyError, TypeError, ValueError) as exc:
        return RerankResult([], False, f"invalid rerank result item: {exc}")

    if len(ranked) != len(candidates):
        return RerankResult(
            [], False,
            f"rerank returned {len(ranked)} valid results for {len(candidates)} candidates",
        )
    return RerankResult(ranked, True)
