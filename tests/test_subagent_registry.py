"""tests/test_subagent_registry.py — SubagentRegistry 容量和清理测试"""

import pytest

from novare.subagents.registry import SubagentRegistry
from novare.subagents.types import SubagentStatus, SubagentType


class TestSubagentRegistryCleanup:
    """M7: create() 自动清理 + max_records 上限"""

    def test_create_triggers_cleanup_finished(self):
        """create() 应自动调用 cleanup_finished，清理过期 finished 记录。"""
        reg = SubagentRegistry(cleanup_age_seconds=0.01)
        r1 = reg.create(SubagentType.GENERAL, "task1")
        reg.complete(r1.subagent_id, "done")
        assert len(reg.list_all()) == 1

        import time
        time.sleep(0.02)  # 超过 cleanup_age_seconds

        r2 = reg.create(SubagentType.GENERAL, "task2")
        # create() 应触发 cleanup，r1 被清理
        assert len(reg.list_all()) == 1
        assert reg.get(r1.subagent_id) is None
        assert reg.get(r2.subagent_id) is not None

    def test_enforce_max_records_evicts_oldest_finished(self):
        """超过 max_records 时，最老的 finished 记录被删除。"""
        reg = SubagentRegistry(max_records=5, cleanup_age_seconds=9999)
        finished_ids = []
        for i in range(5):
            r = reg.create(SubagentType.GENERAL, f"task-{i}")
            reg.complete(r.subagent_id, f"result-{i}")
            finished_ids.append(r.subagent_id)

        assert len(reg.list_all()) == 5
        # 第 6 个 create 应该触发 _enforce_max_records，删除最老的 finished
        r6 = reg.create(SubagentType.GENERAL, "task-5")
        assert len(reg.list_all()) == 5
        assert reg.get(finished_ids[0]) is None  # 最老的被删除
        assert reg.get(r6.subagent_id) is not None

    def test_enforce_max_records_does_not_evict_running(self):
        """全是 active 记录时不删除 active，只记 warning。"""
        reg = SubagentRegistry(max_records=3, cleanup_age_seconds=9999)
        for i in range(3):
            reg.create(SubagentType.GENERAL, f"task-{i}")

        active = reg.list_active()
        assert len(active) == 3

        # 全是 PENDING，不应被删除
        r4 = reg.create(SubagentType.GENERAL, "task-3")
        # 会触发 warning 但不删 active
        all_records = reg.list_all()
        assert len(all_records) == 4  # 没有删除任何 active
        assert reg.get(r4.subagent_id) is not None

    def test_enforce_max_records_prefers_finished_over_running(self):
        """有 running 和 finished 混合时，只删 finished。"""
        reg = SubagentRegistry(max_records=4, cleanup_age_seconds=9999)
        # 创建 3 个 finished
        for i in range(3):
            r = reg.create(SubagentType.GENERAL, f"done-{i}")
            reg.complete(r.subagent_id, f"result-{i}")
        # 创建 1 个 running（不完成）
        running = reg.create(SubagentType.GENERAL, "running-task")

        assert len(reg.list_all()) == 4
        # 第 5 个 create：应删除最老的 finished，保留 running
        reg.create(SubagentType.GENERAL, "new-task")
        assert reg.get(running.subagent_id) is not None
        # 最老的 finished 被删
        assert len(reg.list_all()) == 4

    def test_custom_max_records(self):
        """自定义 max_records 生效。"""
        reg = SubagentRegistry(max_records=2, cleanup_age_seconds=9999)
        r1 = reg.create(SubagentType.GENERAL, "t1")
        reg.complete(r1.subagent_id, "done")
        r2 = reg.create(SubagentType.GENERAL, "t2")
        reg.complete(r2.subagent_id, "done")
        assert len(reg.list_all()) == 2

        # 第 3 个 → 触发 evict
        reg.create(SubagentType.GENERAL, "t3")
        assert len(reg.list_all()) == 2


class TestSubagentRegistryCompat:
    """确认现有 API 仍然兼容。"""

    def test_create_returns_record(self):
        reg = SubagentRegistry()
        r = reg.create(SubagentType.SEARCH, "find papers")
        assert r.subagent_id.startswith("sa-")
        assert r.status == SubagentStatus.PENDING
        assert r.task == "find papers"

    def test_get_and_list(self):
        reg = SubagentRegistry()
        r1 = reg.create(SubagentType.GENERAL, "t1")
        r2 = reg.create(SubagentType.GENERAL, "t2")
        assert reg.get(r1.subagent_id) is not None
        assert len(reg.list_active()) == 2
        assert len(reg.list_all()) == 2

    def test_complete_and_fail(self):
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "t")
        reg.complete(r.subagent_id, "done")
        assert reg.get(r.subagent_id).status == SubagentStatus.COMPLETED

        r2 = reg.create(SubagentType.GENERAL, "t2")
        reg.fail(r2.subagent_id, "oops")
        assert reg.get(r2.subagent_id).status == SubagentStatus.FAILED

    def test_get_output(self):
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "t")
        reg.complete(r.subagent_id, "result text")
        output = reg.get_output(r.subagent_id)
        assert output is not None
        assert output.result == "result text"

    def test_cleanup_finished_with_age(self):
        reg = SubagentRegistry()
        r = reg.create(SubagentType.GENERAL, "t")
        reg.complete(r.subagent_id, "done")
        # 清理 age=0.01 并等一下 → 应该清理
        import time
        time.sleep(0.02)
        count = reg.cleanup_finished(max_age_seconds=0.01)
        assert count == 1
        assert reg.get(r.subagent_id) is None
