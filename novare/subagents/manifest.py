"""novare/subagents/manifest.py — 文件级子智能体清单（调试/审计用）

借鉴 claw-code 的文件清单模式，但简化为写入式日志：
每个子智能体的状态变更写入 workspace/.novare/subagents/{id}.json。
仅用于调试和事后分析，不参与运行时通信。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("novare.subagents.manifest")


def _manifest_dir(workspace: Path) -> Path:
    return workspace / ".novare" / "subagents"


def read_manifest(workspace: Path, subagent_id: str) -> dict | None:
    """读取子智能体清单"""
    path = _manifest_dir(workspace) / f"{subagent_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read manifest %s: %s", path, e)
        return None


def list_manifests(workspace: Path) -> list[dict]:
    """列出所有清单"""
    d = _manifest_dir(workspace)
    if not d.exists():
        return []

    result = []
    for p in sorted(d.glob("sa-*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return result
