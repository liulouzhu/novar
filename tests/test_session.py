"""tests/test_session.py"""

import json
import time
from pathlib import Path

import pytest

from novare.session import Session


class TestSession:
    def test_create_session(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        assert session.session_id is not None
        assert session.messages == []

    def test_add_user_message(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        session.add_user_message("Hello")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"

    def test_add_assistant_message(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        session.add_assistant_message("Hi there")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "assistant"
        assert session.messages[0]["content"] == "Hi there"

    def test_add_assistant_message_with_tool_calls(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "paper_search", "arguments": "{}"}}]
        session.add_assistant_message("", tool_calls=tool_calls)
        msg = session.messages[0]
        assert msg["tool_calls"] == tool_calls

    def test_add_tool_result(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        session.add_tool_result("call_1", "result text")
        msg = session.messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == "result text"

    def test_save_and_load(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        session.add_user_message("Test message")
        session.save()

        loaded = Session.load(session.session_id, workspace=tmp_workspace)
        assert loaded.session_id == session.session_id
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "Test message"

    def test_load_latest(self, tmp_workspace):
        s1 = Session(workspace=tmp_workspace)
        s1.add_user_message("First")
        s1.save()

        # Windows 文件系统 mtime 存在量化粒度：两次保存间隔过短时
        # mtime 可能相同，"latest" 的排序将失去依据 —— 显式拉开间隔
        time.sleep(0.02)

        s2 = Session(workspace=tmp_workspace)
        s2.add_user_message("Second")
        s2.save()

        loaded = Session.load("latest", workspace=tmp_workspace)
        assert loaded.session_id == s2.session_id

    def test_list_sessions(self, tmp_workspace):
        s1 = Session(workspace=tmp_workspace)
        s1.add_user_message("A")
        s1.save()
        s2 = Session(workspace=tmp_workspace)
        s2.add_user_message("B")
        s2.save()

        sessions = Session.list_sessions(workspace=tmp_workspace)
        assert len(sessions) == 2
        assert s1.session_id in sessions
        assert s2.session_id in sessions

    def test_delete_session(self, tmp_workspace):
        session = Session(workspace=tmp_workspace)
        session.add_user_message("Delete me")
        session.save()
        sid = session.session_id

        session.delete()
        assert not (tmp_workspace / ".novare" / "sessions" / f"{sid}.jsonl").exists()
