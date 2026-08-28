"""MinerU API 客户端 — PDF 解析服务"""

import asyncio
import io
import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("research-server.mineru")

MINERU_API = "https://mineru.net/api/v4/extract/task"
MINERU_TOKEN = os.environ.get("MINERU_API_TOKEN", "")


@dataclass
class MinerUResult:
    success: bool
    markdown: str = ""
    saved_dir: str = ""
    error: str = ""


async def parse_pdf_with_mineru(pdf_url: str, token: str | None = None,
                                 save_dir: str | None = None) -> MinerUResult:
    """用 MinerU API 解析 PDF，返回 Markdown 内容

    Args:
        pdf_url: PDF 链接
        token: MinerU API token
        save_dir: 保存目录（可选），会保存 zip 和 markdown
    """
    api_token = token or MINERU_TOKEN
    if not api_token:
        return MinerUResult(success=False, error="MINERU_API_TOKEN not set")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=300) as client:
        # 1. 创建解析任务
        try:
            resp = await client.post(MINERU_API, json={"url": pdf_url}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return MinerUResult(success=False, error=f"Failed to create task: {e}")

        if data.get("code") != 0:
            return MinerUResult(success=False, error=f"API error: {data.get('msg', 'unknown')}")

        task_id = data["data"]["task_id"]
        logger.info("MinerU task created: %s", task_id)

        # 2. 轮询任务状态
        poll_url = f"{MINERU_API}/{task_id}"
        for _ in range(120):  # 最多等 10 分钟
            await asyncio.sleep(5)
            try:
                resp = await client.get(poll_url, headers=headers)
                resp.raise_for_status()
                status = resp.json()
            except Exception as e:
                logger.warning("Poll failed: %s", e)
                continue

            state = status.get("data", {}).get("state", "")
            progress = status.get("data", {}).get("extract_progress", {})
            if progress:
                logger.info("MinerU progress: %s/%s pages",
                           progress.get("extracted_pages", 0),
                           progress.get("total_pages", 0))

            if state == "done":
                zip_url = status["data"].get("full_zip_url")
                if not zip_url:
                    return MinerUResult(success=False, error="No zip URL in response")
                break
            elif state == "failed":
                err = status.get("data", {}).get("err_msg", "unknown error")
                return MinerUResult(success=False, error=f"Task failed: {err}")
        else:
            return MinerUResult(success=False, error="Task timed out")

        # 3. 下载 zip
        try:
            resp = await client.get(zip_url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            zip_bytes = resp.content
        except Exception as e:
            return MinerUResult(success=False, error=f"Failed to download zip: {e}")

        # 4. 保存到本地
        saved_dir = ""
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                zip_path = os.path.join(save_dir, "mineru_output.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_bytes)
                logger.info("Saved MinerU zip: %s", zip_path)
            except Exception as e:
                logger.warning("Failed to save zip: %s", e)

        # 5. 从 zip 中提取 markdown
        try:
            markdown = _extract_markdown_from_zip(zip_bytes)
        except Exception as e:
            return MinerUResult(success=False, error=f"Failed to extract markdown: {e}")

        if not markdown:
            return MinerUResult(success=False, error="No markdown content in zip")

        # 6. 保存 markdown
        if save_dir:
            try:
                md_path = os.path.join(save_dir, "content.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                saved_dir = save_dir
                logger.info("Saved MinerU markdown: %s", md_path)
            except Exception as e:
                logger.warning("Failed to save markdown: %s", e)

        logger.info("MinerU parsing complete: %d chars", len(markdown))
        return MinerUResult(success=True, markdown=markdown, saved_dir=saved_dir)


def _extract_markdown_from_zip(zip_bytes: bytes) -> str:
    """从 MinerU 返回的 zip 中提取 markdown 内容"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # MinerU zip 结构: <task_id>/auto/<filename>.md 或类似
        md_files = [f for f in zf.namelist() if f.endswith(".md")]
        if not md_files:
            # 尝试 .txt
            md_files = [f for f in zf.namelist() if f.endswith(".txt")]
        if not md_files:
            # 尝试任何文本文件
            md_files = [f for f in zf.namelist()
                       if not f.endswith((".json", ".png", ".jpg", ".pdf"))]

        if not md_files:
            return ""

        # 按文件名排序，取最长的（通常是完整内容）
        md_files.sort(key=len, reverse=True)

        parts = []
        for md_file in md_files:
            content = zf.read(md_file).decode("utf-8", errors="replace")
            parts.append(content)

        return "\n\n".join(parts)
