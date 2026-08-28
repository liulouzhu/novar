"""Tests for the deterministic task-completion evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "task_completion.py"
SPEC = importlib.util.spec_from_file_location("task_completion", MODULE_PATH)
assert SPEC and SPEC.loader
task_completion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = task_completion
SPEC.loader.exec_module(task_completion)


def test_evaluate_requires_every_check_to_pass(tmp_path):
    (tmp_path / "answer.txt").write_text("done", encoding="utf-8")
    task = {
        "id": "example",
        "input": "do the task",
        "checks": [
            {"type": "answer_contains", "text": "complete"},
            {"type": "tool_succeeded", "name": "write_file"},
            {"type": "file_contains", "path": "answer.txt", "text": "done"},
        ],
    }
    result = {
        "answer": "Task complete",
        "tool_events": [{"event": "end", "name": "write_file"}],
        "error": "",
        "metadata": {},
    }

    checks = task_completion._evaluate(task, result, tmp_path)

    assert all(check.passed for check in checks)


def test_file_check_cannot_escape_workspace(tmp_path):
    result = {"answer": "", "tool_events": [], "error": "", "metadata": {}}
    with pytest.raises(task_completion.TaskValidationError, match="escapes task workspace"):
        task_completion._run_check({"type": "file_exists", "path": "../outside.txt"}, result, tmp_path)


def test_fixtures_preserve_relative_paths(tmp_path, monkeypatch):
    fixture = Path("data") / "task-completion-fixture.txt"
    fake_repo = tmp_path / "repo"
    monkeypatch.setattr(task_completion, "REPO_ROOT", fake_repo)
    source = fake_repo / fixture
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("fixture", encoding="utf-8")
    workspace = tmp_path / "workspace"
    task_completion._prepare_workspace(
        {"id": "fixture-task", "input": "use fixture", "fixtures": [str(fixture)]},
        workspace,
    )
    assert (workspace / fixture).read_text(encoding="utf-8") == "fixture"


def test_summary_reports_sr_and_pass_at_k():
    summary = task_completion._summarise([
        {"task_id": "a", "passed": False, "duration_seconds": 1.0, "error": "first failure"},
        {"task_id": "a", "passed": True, "duration_seconds": 2.0, "error": ""},
        {"task_id": "b", "passed": False, "duration_seconds": 3.0, "error": "second failure"},
    ], runs_per_task=2)

    assert summary["attempt_success_rate"] == pytest.approx(1 / 3)
    assert summary["sr_at_1"] == pytest.approx(0.0)
    assert summary["pass_at_k"] == pytest.approx(1 / 2)
    assert summary["p95_duration_seconds"] > 2.0
