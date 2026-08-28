"""tests/test_sandbox_lifecycle.py — sandbox lifecycle management tests"""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# restricted_open — pathlib.Path support
# ---------------------------------------------------------------------------

class TestRestrictedOpen:
    """executor.restricted_open must accept both str and pathlib.Path."""

    def _import_restricted_open(self):
        """Import restricted_open from the sandbox executor without running main()."""
        import importlib
        import sys

        # The executor lives outside the normal package tree; ensure its dir is on sys.path
        executor_dir = str(Path(__file__).resolve().parent.parent / "docker" / "sandbox")
        if executor_dir not in sys.path:
            sys.path.insert(0, executor_dir)

        # Fresh import each time so builtins mutations don't leak
        if "executor" in sys.modules:
            importlib.reload(sys.modules["executor"])
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("executor", os.path.join(executor_dir, "executor.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["executor"] = mod

        return sys.modules["executor"].restricted_open

    def test_str_path_allowed(self, tmp_path):
        restricted_open = self._import_restricted_open()
        # Patch ALLOWED_PATHS to include our tmp_path for testing
        import sys
        executor = sys.modules["executor"]
        allowed = ("/data/", "/output/")
        # We can't easily test with real /data/ paths; just verify the function
        # doesn't crash on str input and respects the check
        with pytest.raises(PermissionError, match="not in allowed paths"):
            restricted_open(str(tmp_path / "test.txt"))

    def test_pathlib_path_allowed(self, tmp_path):
        restricted_open = self._import_restricted_open()
        # pathlib.Path should not cause AttributeError
        with pytest.raises(PermissionError, match="not in allowed paths"):
            restricted_open(tmp_path / "test.txt")

    def test_pathlib_path_no_startswith_error(self, tmp_path):
        """Ensure pathlib.Path doesn't trigger 'startswith' AttributeError."""
        restricted_open = self._import_restricted_open()
        p = Path("/data/test.txt")
        # Should raise PermissionError (path check), NOT AttributeError
        try:
            restricted_open(p)
        except PermissionError:
            pass  # expected — the important thing is no AttributeError
        except AttributeError:
            pytest.fail("restricted_open raised AttributeError on pathlib.Path")


# ---------------------------------------------------------------------------
# DockerSandboxManager — cleanup / shutdown / labels
# ---------------------------------------------------------------------------

class TestSandboxManagerLifecycle:
    """Test startup cleanup, idle cleanup, shutdown, and label handling."""

    def _make_manager(self):
        """Create a DockerSandboxManager with a mocked Docker client."""
        from web.backend.sandbox.manager import DockerSandboxManager

        manager = DockerSandboxManager.__new__(DockerSandboxManager)
        manager.client = MagicMock()
        manager._containers = {}
        manager._last_used = {}
        return manager

    def test_startup_removes_labeled_containers(self):
        from web.backend.sandbox.manager import SANDBOX_LABELS

        manager = self._make_manager()
        mock_container = MagicMock()
        mock_container.name = "sandbox-old123"
        manager.client.containers.list.return_value = [mock_container]

        manager.startup()

        manager.client.containers.list.assert_called_once_with(
            all=True,
            filters={"label": [f"{k}={v}" for k, v in SANDBOX_LABELS.items()]},
        )
        mock_container.remove.assert_called_once_with(force=True)

    def test_startup_handles_docker_errors_gracefully(self):
        manager = self._make_manager()
        manager.client.containers.list.side_effect = RuntimeError("docker down")

        # Should not raise
        manager.startup()

    def test_cleanup_idle_destroys_expired(self):
        manager = self._make_manager()
        manager._containers["u1"] = MagicMock()
        manager._last_used["u1"] = time.time() - 99999  # well past IDLE_TIMEOUT

        manager.cleanup_idle()

        assert "u1" not in manager._containers
        assert "u1" not in manager._last_used

    def test_cleanup_idle_keeps_fresh(self):
        manager = self._make_manager()
        manager._containers["u2"] = MagicMock()
        manager._last_used["u2"] = time.time()  # just used

        manager.cleanup_idle()

        assert "u2" in manager._containers

    def test_shutdown_destroys_all(self):
        manager = self._make_manager()
        c1, c2 = MagicMock(), MagicMock()
        manager._containers = {"a": c1, "b": c2}
        manager._last_used = {"a": time.time(), "b": time.time()}

        manager.shutdown()

        c1.remove.assert_called_once_with(force=True)
        c2.remove.assert_called_once_with(force=True)
        assert manager._containers == {}
        assert manager._last_used == {}

    def test_container_creation_passes_labels(self):
        """Verify that get_or_create passes SANDBOX_LABELS to containers.run."""
        from docker.errors import NotFound
        from web.backend.sandbox.manager import SANDBOX_LABELS

        manager = self._make_manager()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.exec_run.return_value = MagicMock(exit_code=0)
        manager.client.containers.get.side_effect = NotFound("not found")
        manager.client.containers.run.return_value = mock_container
        manager.client.images.get.return_value = MagicMock(id="img1")
        mock_container.image.id = "img1"

        with patch.object(type(manager), "_has_writable_runtime_dirs", return_value=True), \
             patch("web.backend.sandbox.manager.os.makedirs"), \
             patch("web.backend.sandbox.manager.os.path.abspath", return_value="/fake/project"):
            manager.get_or_create("test-user-12345")

        call_kwargs = manager.client.containers.run.call_args
        assert call_kwargs[1]["labels"] == SANDBOX_LABELS


# ---------------------------------------------------------------------------
# app.py lifespan — idle cleanup task & shutdown hook
# ---------------------------------------------------------------------------

class TestAppLifespan:
    """Verify that lifespan wires up the idle cleanup task and sandbox shutdown."""

    @pytest.mark.asyncio
    async def test_lifespan_cancels_cleanup_task_and_shuts_down_sandbox(self):
        """On shutdown the cleanup task is cancelled and sandbox_manager.shutdown is called."""
        from web.backend import app as app_module

        mock_sandbox = MagicMock()
        mock_agent_service = MagicMock()
        mock_agent_service.initialize = AsyncMock()
        mock_agent_service.shutdown = AsyncMock()

        shutdown_called = []
        mock_sandbox.shutdown = MagicMock(side_effect=lambda: shutdown_called.append(True))

        with patch.object(app_module, "sandbox_manager", mock_sandbox), \
             patch.object(app_module, "agent_service", mock_agent_service), \
             patch("web.backend.app.Base") as mock_base:

            cm = app_module.lifespan(MagicMock())
            await cm.__aenter__()
            # Shutdown
            await cm.__aexit__(None, None, None)

        assert len(shutdown_called) == 1, "sandbox_manager.shutdown() should be called exactly once"
        mock_agent_service.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_cleanup_task_is_awaitable(self):
        """The idle cleanup background task starts and can be cancelled."""
        from web.backend import app as app_module

        mock_sandbox = MagicMock()
        mock_sandbox.startup = MagicMock()
        mock_sandbox.shutdown = MagicMock()
        mock_sandbox.cleanup_idle = MagicMock()

        mock_agent_service = MagicMock()
        mock_agent_service.initialize = AsyncMock()
        mock_agent_service.shutdown = AsyncMock()

        with patch.object(app_module, "sandbox_manager", mock_sandbox), \
             patch.object(app_module, "agent_service", mock_agent_service), \
             patch("web.backend.app.Base") as mock_base:

            cm = app_module.lifespan(MagicMock())
            await cm.__aenter__()

            mock_sandbox.startup.assert_called_once()

            await cm.__aexit__(None, None, None)

        mock_sandbox.shutdown.assert_called_once()
