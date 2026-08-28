"""novare/tools/file_ops.py -- 文件操作工具"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path


def _resolve_workspace_path(raw_path: str | Path, workspace: Path) -> Path:
    """将用户传入的路径解析为 workspace 内的绝对路径，拒绝逃逸。

    - 相对路径：基于 workspace 解析
    - 绝对路径：必须在 workspace 内
    - .. 逃逸和符号链接逃逸均被拒绝
    """
    workspace_root = workspace.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if resolved == workspace_root:
        return resolved
    # workspace_root 必须是 resolved 的祖先
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise ValueError(f"Path outside workspace: {raw_path}")
    return resolved


async def handle_read_file(args: dict, workspace: Path = Path(".")) -> str:
    try:
        path = _resolve_workspace_path(args["path"], workspace)
    except ValueError as e:
        return f"Error: {e}"
    if not path.exists():
        return f"Error: File not found: {args['path']}"
    if path.is_dir():
        return f"Error: Is a directory: {args['path']}"
    try:
        content = path.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"Error reading file: {e}"


async def handle_write_file(args: dict, workspace: Path = Path(".")) -> str:
    try:
        path = _resolve_workspace_path(args["path"], workspace)
    except ValueError as e:
        return f"Error: {e}"
    content = args["content"]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"OK: Written {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def handle_edit_file(args: dict, workspace: Path = Path(".")) -> str:
    try:
        path = _resolve_workspace_path(args["path"], workspace)
    except ValueError as e:
        return f"Error: {e}"
    old_string = args["old_string"]
    new_string = args["new_string"]

    if not path.exists():
        return f"Error: File not found: {args['path']}"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    if old_string not in content:
        return f"Error: old_string not found in {args['path']}"

    count = content.count(old_string)
    new_content = content.replace(old_string, new_string, 1)

    try:
        path.write_text(new_content, encoding="utf-8")
        extra = f" ({count - 1} remaining)" if count > 1 else ""
        return f"OK: Replaced old_string in {args['path']}{extra}"
    except Exception as e:
        return f"Error writing file: {e}"


async def handle_glob_search(args: dict, workspace: Path = Path(".")) -> str:
    pattern = args["pattern"]
    try:
        search_path = _resolve_workspace_path(args.get("path", "."), workspace)
    except ValueError as e:
        return f"Error: {e}"

    if not search_path.exists():
        return f"Error: Path not found: {args.get('path', '.')}"

    matches = []
    for root, dirs, files in os.walk(search_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, search_path)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f, pattern):
                matches.append(rel)

    if not matches:
        return "No files found matching pattern."

    matches.sort()
    return "\n".join(matches)


async def handle_grep_search(args: dict, workspace: Path = Path(".")) -> str:
    pattern = args["pattern"]
    try:
        search_path = _resolve_workspace_path(args.get("path", "."), workspace)
    except ValueError as e:
        return f"Error: {e}"
    glob_filter = args.get("glob", None)

    if not search_path.exists():
        return f"Error: Path not found: {args.get('path', '.')}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    results = []
    for root, dirs, files in os.walk(search_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if glob_filter and not fnmatch.fnmatch(f, glob_filter):
                continue
            full = os.path.join(root, f)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = os.path.relpath(full, search_path)
                        results.append(f"{rel}:{i}: {line.strip()}")
            except Exception:
                continue

    if not results:
        return "No matches found."

    return "\n".join(results[:50])
