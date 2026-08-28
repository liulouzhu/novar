"""论文 ID 规范化工具

统一各种 arXiv ID 格式为 canonical form: arxiv:<YYYY>.<NNNNN>

支持的输入格式:
  - 裸 ID: 2308.11681
  - 前缀 ID: arxiv:2308.11681
  - arXiv URL: https://arxiv.org/abs/2308.11681, https://arxiv.org/pdf/2308.11681
  - 带版本号: 2308.11681v2, arxiv:2308.11681v2
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

# arXiv ID pattern: YYYY.NNNNN (may have optional version suffix)
_ARXIV_ID_RE = re.compile(
    r"^(?:(?:https?://(?:arxiv\.org/(?:abs|pdf|html)/))?)"
    r"(?:(?:arxiv:)?)"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)"
    r"(?:\.pdf)?(?:[?#].*)?"
    r"$",
    re.IGNORECASE,
)

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_ARXIV_IN_TEXT_RE = re.compile(r"(?:arxiv\s*:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


def canonicalize_paper_id(paper_id: str) -> str:
    """将各种格式的论文 ID 规范化为 canonical form。

    arXiv 论文 → arxiv:YYYY.NNNNN（去掉版本号）
    其他 ID → 原样返回（doi:..., s2:..., parsed 等）

    Examples:
        canonicalize_paper_id("2308.11681") → "arxiv:2308.11681"
        canonicalize_paper_id("arxiv:2308.11681") → "arxiv:2308.11681"
        canonicalize_paper_id("arxiv:2308.11681v2") → "arxiv:2308.11681"
        canonicalize_paper_id("https://arxiv.org/abs/2308.11681") → "arxiv:2308.11681"
        canonicalize_paper_id("https://arxiv.org/pdf/2308.11681v3") → "arxiv:2308.11681"
        canonicalize_paper_id("doi:10.1234/abc") → "doi:10.1234/abc"
        canonicalize_paper_id("s2:12345") → "s2:12345"
    """
    if not paper_id:
        return paper_id

    pid = paper_id.strip()
    m = _ARXIV_ID_RE.match(pid)
    if m:
        arxiv_id = m.group(1)
        # 去掉版本号
        base = re.sub(r"v\d+$", "", arxiv_id)
        return f"arxiv:{base}"

    doi_candidate = unquote(pid)
    doi_candidate = re.sub(
        r"^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        doi_candidate,
        flags=re.IGNORECASE,
    ).strip()
    doi_match = _DOI_RE.fullmatch(doi_candidate.rstrip(".,;"))
    if doi_match:
        return f"doi:{doi_match.group(0).lower()}"

    if pid.lower().startswith("s2:"):
        return f"s2:{pid[3:].strip()}"

    # 非标准 ID 原样返回
    return pid


def is_arxiv_id(paper_id: str) -> bool:
    """判断是否为 arXiv 论文 ID（任何形式）"""
    if not paper_id:
        return False
    return _ARXIV_ID_RE.match(paper_id.strip()) is not None


def identifier_type(identifier: str) -> str:
    prefix, separator, _ = identifier.partition(":")
    return prefix.lower() if separator and prefix.lower() in {"doi", "arxiv", "s2"} else "other"


def canonicalize_identifiers(identifiers: list[str]) -> list[str]:
    result: list[str] = []
    for identifier in identifiers:
        canonical = canonicalize_paper_id(identifier)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def extract_document_identifiers(text: str, *, scan_chars: int = 8000) -> list[str]:
    """Extract likely document identifiers from the front matter, not references."""
    front_matter = text[:scan_chars]
    references = re.search(r"(?im)^#{0,3}\s*(?:references|bibliography)\s*$", front_matter)
    if references:
        front_matter = front_matter[:references.start()]

    identifiers: list[str] = []
    for match in _DOI_RE.finditer(front_matter):
        identifiers.append(f"doi:{match.group(0).rstrip('.,;').lower()}")

    for line in front_matter.splitlines():
        if "arxiv" not in line.lower():
            continue
        match = _ARXIV_IN_TEXT_RE.search(line)
        if match:
            arxiv_base = re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
            identifiers.append(f"arxiv:{arxiv_base}")

    canonical = canonicalize_identifiers(identifiers)
    return sorted(canonical, key=lambda item: ({"doi": 0, "arxiv": 1}.get(identifier_type(item), 9), item))
