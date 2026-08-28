"""novare/skill.py — Skill 发现、解析、加载

Skill 是 prompt 模板（markdown 文件 + YAML frontmatter）。
调用 skill = 将模板内容作为 user message 发送给 LLM。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("novare.skill")

# Frontmatter 正则：匹配 --- 开头和结尾之间的内容
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# 简单 YAML 解析：只取 key: value 行
_YAML_KV_RE = re.compile(r"^(\w[\w-]*)\s*:\s*(.+)$", re.MULTILINE)


@dataclass
class Skill:
    """一个已加载的 skill"""
    name: str                # 调用名（文件名，不含 .md）
    description: str         # frontmatter 中的 description
    prompt_template: str     # markdown 正文（不含 frontmatter）
    source: Path             # 文件路径

    def render(self, arguments: str = "") -> str:
        """将模板渲染为最终 prompt，替换 $ARGUMENTS 占位符"""
        return self.prompt_template.replace("$ARGUMENTS", arguments).strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 YAML frontmatter，返回 (metadata_dict, body)"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw_yaml = m.group(1)
    body = text[m.end():]

    meta: dict[str, str] = {}
    for km in _YAML_KV_RE.finditer(raw_yaml):
        key = km.group(1).strip()
        val = km.group(2).strip().strip("\"'")
        meta[key] = val

    return meta, body


def _load_skill_file(path: Path) -> Skill | None:
    """从单个 .md 文件加载一个 skill"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot read skill file %s: %s", path, e)
        return None

    meta, body = _parse_frontmatter(text)

    # 名称：优先取 frontmatter 的 name，否则用文件名
    name = meta.get("name", "")
    if not name:
        # 目录式：取父目录名；扁平式：取文件 stem
        if path.name.lower() == "skill.md":
            name = path.parent.name
        else:
            name = path.stem

    description = meta.get("description", "")

    # 正文为空则跳过
    body = body.strip()
    if not body:
        logger.warning("Skill %s has empty body, skipping", name)
        return None

    return Skill(
        name=name,
        description=description,
        prompt_template=body,
        source=path,
    )


def discover_skills(roots: list[Path]) -> list[Skill]:
    """从多个根目录发现所有 skill。

    扫描两种布局：
      - .novare/skills/<name>/SKILL.md  （目录式）
      - .novare/skills/<name>.md        （扁平式）

    同名 skill 按 roots 顺序 shadowing（先发现的优先）。
    """
    seen_names: set[str] = set()
    skills: list[Skill] = []

    for root in roots:
        if not root.is_dir():
            continue

        # 扫描目录式：<name>/SKILL.md
        for child in sorted(root.iterdir()):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.is_file():
                    skill = _load_skill_file(skill_file)
                    if skill and skill.name not in seen_names:
                        seen_names.add(skill.name)
                        skills.append(skill)

        # 扫描扁平式：<name>.md（排除 SKILL.md 本身）
        for md_file in sorted(root.glob("*.md")):
            if md_file.name.lower() == "skill.md":
                continue
            skill = _load_skill_file(md_file)
            if skill and skill.name not in seen_names:
                seen_names.add(skill.name)
                skills.append(skill)

    logger.info("Discovered %d skills from %d roots", len(skills), len(roots))
    return skills
