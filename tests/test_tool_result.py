"""tests/test_tool_result.py — PR 1：tool result 解析向后兼容

验证旧 JSON、普通文本、"Error..." 前缀格式仍可解析，
且新字段 error_code / retryable / outcome / attempts 正确提取。
"""

import json

from novare.tool_result import parse_tool_result


class TestLegacyFormats:
    """旧格式仍可解析（PR 1 不破坏现有消费方）。"""

    def test_old_json_ok_true(self):
        raw = json.dumps({
            "ok": True,
            "summary": "found 3 papers",
            "data": [{"id": "p1"}],
            "sources": ["arXiv"],
            "providers": ["arxiv"],
        })
        parsed = parse_tool_result(raw)
        assert parsed.ok is True
        assert parsed.summary == "found 3 papers"
        assert parsed.data == [{"id": "p1"}]
        assert parsed.is_json is True
        # 新字段取默认值
        assert parsed.error_code is None
        assert parsed.retryable is False
        assert parsed.outcome == "not_applied"
        assert parsed.attempts == 1

    def test_old_json_ok_false(self):
        raw = json.dumps({"ok": False, "error": "something bad"})
        parsed = parse_tool_result(raw)
        assert parsed.ok is False
        assert parsed.error == "something bad"
        assert parsed.error_code is None
        assert parsed.retryable is False

    def test_plain_text(self):
        parsed = parse_tool_result("some plain result text")
        assert parsed.ok is True
        assert parsed.is_json is False
        assert parsed.error is None
        assert parsed.error_code is None

    def test_error_prefix_text(self):
        parsed = parse_tool_result("Error: File not found: /x")
        assert parsed.ok is False
        assert parsed.is_json is False
        assert parsed.error == "Error: File not found: /x"

    def test_chinese_error_prefix(self):
        parsed = parse_tool_result("错误：网络异常")
        assert parsed.ok is False

    def test_search_failed_prefix(self):
        parsed = parse_tool_result("搜索失败：无结果")
        assert parsed.ok is False


class TestNewStructuredFormat:
    """PR 1 统一结构化错误格式的解析。"""

    def test_full_structured_failure(self):
        raw = json.dumps({
            "ok": False,
            "error": "Error executing paper_search: upstream timeout",
            "error_code": "UPSTREAM_TIMEOUT",
            "retryable": True,
            "outcome": "retry_exhausted",
            "attempts": 3,
        })
        parsed = parse_tool_result(raw)
        assert parsed.ok is False
        assert parsed.error == "Error executing paper_search: upstream timeout"
        assert parsed.error_code == "UPSTREAM_TIMEOUT"
        assert parsed.retryable is True
        assert parsed.outcome == "retry_exhausted"
        assert parsed.attempts == 3

    def test_partial_structured_failure_defaults(self):
        """缺少部分新字段时取默认值，不崩溃。"""
        raw = json.dumps({"ok": False, "error": "boom", "error_code": "BAD_REQUEST"})
        parsed = parse_tool_result(raw)
        assert parsed.error_code == "BAD_REQUEST"
        assert parsed.retryable is False
        assert parsed.outcome == "not_applied"
        assert parsed.attempts == 1

    def test_attempts_non_int_falls_back(self):
        raw = json.dumps({"ok": False, "error": "x", "attempts": "three"})
        parsed = parse_tool_result(raw)
        assert parsed.attempts == 1

    def test_attempts_zero_is_honored(self):
        raw = json.dumps({"ok": False, "error": "x", "attempts": 0})
        parsed = parse_tool_result(raw)
        assert parsed.attempts == 0

    def test_structured_ok_true_keeps_defaults(self):
        raw = json.dumps({"ok": True, "summary": "s"})
        parsed = parse_tool_result(raw)
        assert parsed.ok is True
        assert parsed.error_code is None
