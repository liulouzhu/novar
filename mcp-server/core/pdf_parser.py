"""PDF 内容提取模块 - pymupdf4llm 转 Markdown，按章节分块"""

import logging
import os
import re

logger = logging.getLogger("research-server.pdf_parser")

# 章节标题模式
SECTION_PATTERNS = [
    re.compile(r"^#+\s+(.+)$", re.MULTILINE),                    # Markdown headers
    re.compile(r"^(\d+\.?\s+(?:Introduction|Abstract|Related Work|Methods?|"
               r"Results?|Discussion|Conclusion|References|Acknowledgments|"
               r"Experiments?|Evaluation|Analysis|Background|Approach|"
               r"Proposed|Framework|Model|Dataset|Appendix).*)$",
               re.MULTILINE | re.IGNORECASE),
]

# 分块参数
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120

_STRONG_BREAKS = ["\n\n", "\n", "。", "！", "？", ". "]
_WEAK_BREAKS = ["；", ";", "：", ":", "，", ","]


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """将 PDF 转换为 Markdown 格式"""
    try:
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text
    except ImportError:
        logger.error("pymupdf4llm not installed")
        raise
    except Exception as e:
        logger.error("Failed to parse PDF %s: %s", pdf_path, e)
        # Fallback: 使用 pymupdf 直接提取文本
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e2:
            logger.error("Fallback PDF extraction also failed: %s", e2)
            raise


def split_into_sections(markdown_text: str) -> list[dict]:
    """按章节标题分割 Markdown 文本"""
    sections = []
    lines = markdown_text.split("\n")
    current_section = "abstract"
    current_text = []

    for line in lines:
        # 检查是否是章节标题
        is_header = False
        for pattern in SECTION_PATTERNS:
            match = pattern.match(line.strip())
            if match:
                # 保存当前章节
                text = "\n".join(current_text).strip()
                if text:
                    sections.append({
                        "section": current_section,
                        "text": text,
                    })

                # 开始新章节
                header = match.group(1).strip() if match.lastindex else line.strip()
                current_section = _normalize_section_name(header)
                current_text = []
                is_header = True
                break

        if not is_header:
            current_text.append(line)

    # 保存最后一个章节
    text = "\n".join(current_text).strip()
    if text:
        sections.append({
            "section": current_section,
            "text": text,
        })

    return sections


def _normalize_section_name(header: str) -> str:
    """规范化章节名称"""
    header_lower = header.lower().strip()
    # 去除数字前缀
    header_lower = re.sub(r"^\d+\.?\s*", "", header_lower)

    mapping = {
        "abstract": "abstract",
        "introduction": "introduction",
        "related work": "related_work",
        "background": "background",
        "method": "methods",
        "methods": "methods",
        "methodology": "methods",
        "approach": "methods",
        "proposed method": "methods",
        "model": "methods",
        "framework": "methods",
        "experiment": "experiments",
        "experiments": "experiments",
        "evaluation": "experiments",
        "results": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "acknowledgments": "acknowledgments",
        "acknowledgements": "acknowledgments",
        "references": "references",
        "appendix": "appendix",
    }

    for key, value in mapping.items():
        if key in header_lower:
            return value

    return header_lower.replace(" ", "_")[:50]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """将文本分块，支持重叠"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # 优先在句末强边界断开；失败后再退到弱边界，最后才硬切。
        if end < len(text):
            search_start = max(start + chunk_size // 2, start)
            best_break = _find_chunk_break(text, search_start, end, _STRONG_BREAKS)
            if best_break <= start:
                best_break = _find_chunk_break(text, search_start, end, _WEAK_BREAKS)
            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end < len(text) else end

    return chunks


def _find_chunk_break(text: str, start: int, end: int, separators: list[str]) -> int:
    """在指定窗口内找最靠后的可接受断点。"""
    best_break = -1
    for sep in separators:
        pos = text.rfind(sep, start, end)
        if pos != -1:
            candidate = pos + len(sep)
            if candidate > best_break:
                best_break = candidate
    return best_break


def extract_references(markdown_text: str) -> list[str]:
    """从 Markdown 文本中提取参考文献列表"""
    # 查找 References 章节
    ref_patterns = [
        re.compile(r"(?:^|\n)#+\s*(?:References?|Bibliography)\s*\n(.*?)(?:\n#+\s|\Z)",
                   re.IGNORECASE | re.DOTALL),
        re.compile(r"\nReferences?\n(.*?)(?:\n\n\n|\Z)", re.IGNORECASE | re.DOTALL),
    ]

    ref_text = ""
    for pattern in ref_patterns:
        match = pattern.search(markdown_text)
        if match:
            ref_text = match.group(1).strip()
            break

    if not ref_text:
        return []

    # 按行分割，过滤空行
    refs = []
    for line in ref_text.split("\n"):
        line = line.strip()
        if line and len(line) > 10:  # 过滤太短的行
            refs.append(line)

    return refs


def extract_paper_ids_from_refs(refs: list[str]) -> list[dict]:
    """从参考文献文本中提取 DOI 和 arXiv ID"""
    results = []
    doi_pattern = re.compile(r"(10\.\d{4,}/[^\s]+)")
    arxiv_pattern = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)

    for ref in refs:
        entry = {"raw": ref, "doi": None, "arxiv_id": None}

        doi_match = doi_pattern.search(ref)
        if doi_match:
            entry["doi"] = doi_match.group(1).rstrip(".,;)")

        arxiv_match = arxiv_pattern.search(ref)
        if arxiv_match:
            entry["arxiv_id"] = arxiv_match.group(1)

        if entry["doi"] or entry["arxiv_id"]:
            results.append(entry)

    return results
