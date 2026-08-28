"""novare/reflexion/progress.py — 确定性进展信号（digest 化）与 no-progress 检测

- 每个真实进展信号都转为固定长度 64-hex digest：
  - TaskState.completed item      → kind="completed"，content 哈希
  - TaskState.key finding         → kind="finding"，content 哈希
  - 成功工具 observation          → kind="tool_success"，
    digest = SHA-256(canonical({"kind","tool","action_fingerprint","summary_digest"}))，
    其中 summary_digest = SHA-256(summary bytes)，绝不保存 summary 明文
- ProgressTracker 只基于持久化的 digest 集合（progress_signal_digests）计算
  fingerprint，不再直接把临时 TaskState 列表作为 fingerprint 输入。
- 指纹可仅凭持久化状态完整重建：fingerprint = SHA-256(canonical(sorted(digests)))。
- pending 文本重写、失败、重试、合成失败、forbidden block、Reflection 自身不算进展。
- 相同信号重复出现不算新进展（集合去重）。
- digest 集合有上限（MAX_PROGRESS_SIGNAL_DIGESTS）；超限时使用确定性、
  可审计策略（按字典序移除一项）保持有界。
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

MAX_PROGRESS_SIGNAL_DIGESTS = 512


def _canonical_json(obj) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def progress_signal_digest(
    *,
    kind: str,
    text: str | None = None,
    tool: str | None = None,
    action_fingerprint: str | None = None,
    summary_digest: str | None = None,
) -> str:
    """把一条真实进展信号转为固定长度 64 位小写 hex digest。

    kind ∈ {"completed", "finding", "tool_success"}：
    - completed / finding：digest(canonical({"kind": kind, "text_digest": SHA-256(text)}))
    - tool_success：digest(canonical({
        "kind": "tool_success", "tool": tool,
        "action_fingerprint": action_fingerprint,
        "summary_digest": summary_digest,   # = SHA-256(summary bytes)
      }))
    参与计算的明文（text / summary）不进入结果，也不进入持久化状态。
    """
    if kind == "tool_success":
        payload = {
            "kind": "tool_success",
            "tool": tool or "",
            "action_fingerprint": action_fingerprint or "",
            "summary_digest": summary_digest or "",
        }
    else:
        payload = {
            "kind": kind,
            "text_digest": _sha256_hex(text or ""),
        }
    return _sha256_hex(_canonical_json(payload))


def compute_progress_fingerprint(*, signal_digests: Iterable[str]) -> str:
    """由持久化的 digest 集合计算确定性进展指纹（64-hex）。

    只依赖 progress_signal_digests，可仅凭持久化状态完整重建。
    """
    return _sha256_hex(_canonical_json(sorted(set(signal_digests))))


def _cap_digest_set(digests: set[str], limit: int) -> set[str]:
    """确定性、可审计的上限策略：超出时按字典序移除最小的一项，保持有界。"""
    while len(digests) > limit:
        digests.remove(min(digests))
    return digests


class ProgressTracker:
    """跟踪 no-progress 计数（单调累计 digest 化的进展信号）。

    用法：每个 Agent iteration 结束后调用 update()，
    返回 True 表示本次迭代出现真实进展（no_progress_count 归零），
    False 表示无进展（计数 +1）。

    支持跨进程恢复：from_state() 用 ReflexionState 的 progress_signal_digests 与
    last_progress_fingerprint 初始化；恢复后第一次 update 与恢复指纹比较
    （而非自动视为基线），未产生真实进展时 no_progress_count 继续累加。
    """

    def __init__(
        self,
        initial_signal_digests: Iterable[str] | None = None,
        initial_last_progress_fingerprint: str | None = None,
    ) -> None:
        self._signal_digests: set[str] = _cap_digest_set(
            set(initial_signal_digests or []), MAX_PROGRESS_SIGNAL_DIGESTS,
        )
        self._last_progress_fingerprint = initial_last_progress_fingerprint

    @classmethod
    def from_state(cls, state) -> "ProgressTracker":
        """从 ReflexionState 恢复（digest 集合 + 上次进展指纹）。"""
        return cls(
            initial_signal_digests=getattr(state, "progress_signal_digests", None),
            initial_last_progress_fingerprint=getattr(state, "last_progress_fingerprint", None),
        )

    def sync_to_state(self, state) -> None:
        """把当前 digest 集合与进展指纹同步回 ReflexionState（持久化用）。"""
        state.progress_signal_digests = set(self._signal_digests)
        state.last_progress_fingerprint = self._last_progress_fingerprint

    @property
    def signal_digests(self) -> set[str]:
        """当前累计的进展信号 digest 集合（只读视图，仅 64-hex）。"""
        return set(self._signal_digests)

    @property
    def last_progress_fingerprint(self) -> str | None:
        """最后一次出现真实进展时的累计指纹（正式只读属性）。"""
        return self._last_progress_fingerprint

    def update(
        self,
        *,
        completed: Iterable[str] = (),
        key_findings: Iterable[str] = (),
        success_signal_digests: Iterable[str] = (),
    ) -> bool:
        """更新累计进展信号。返回 True 表示出现真实进展。

        completed / key_findings: TaskState 的真实进展文本（内部转 digest，
          明文不进入状态）；success_signal_digests: 调用方已用
          progress_signal_digest(kind="tool_success", ...) 计算的 64-hex digest。
        同一信号重复出现不算新进展；集合有界（确定性裁剪）。
        """
        for item in completed:
            self._signal_digests.add(progress_signal_digest(kind="completed", text=str(item)))
        for finding in key_findings:
            self._signal_digests.add(progress_signal_digest(kind="finding", text=str(finding)))
        for digest in success_signal_digests:
            if isinstance(digest, str) and digest:
                self._signal_digests.add(digest)
        _cap_digest_set(self._signal_digests, MAX_PROGRESS_SIGNAL_DIGESTS)

        fp = compute_progress_fingerprint(signal_digests=self._signal_digests)
        # 与上次指纹比较（恢复后首次 update 与恢复指纹比较，而非自动视为基线）
        if self._last_progress_fingerprint is not None and fp == self._last_progress_fingerprint:
            return False  # 无进展（digest 集合未变）
        self._last_progress_fingerprint = fp
        return True  # 首次调用（无基线）或集合变化即进展
