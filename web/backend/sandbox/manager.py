import asyncio
import os
import time
import io
import tarfile
import logging
import docker
from docker.errors import NotFound

logger = logging.getLogger(__name__)

IMAGE = "research-sandbox:latest"
IDLE_TIMEOUT = 1800  # 30 minutes

# Labels applied to every sandbox container for identification & cleanup
SANDBOX_LABELS = {
    "app": "research-agent",
    "component": "sandbox",
    "managed-by": "web-backend",
}


class DockerSandboxManager:
    """Manages per-user Docker sandboxes for code execution."""

    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.warning("Docker not available: %s", e)
            self.client = None
        self._containers: dict[str, docker.models.containers.Container] = {}
        self._last_used: dict[str, float] = {}

    def startup(self):
        """Clean up stale sandbox containers left over from previous runs."""
        if not self.client:
            return
        try:
            stale = self.client.containers.list(
                all=True,
                filters={"label": [f"{k}={v}" for k, v in SANDBOX_LABELS.items()]},
            )
            for container in stale:
                try:
                    container.remove(force=True)
                    logger.info("Removed stale sandbox container: %s", container.name)
                except Exception as e:
                    logger.warning("Failed to remove stale sandbox %s: %s", container.name, e)
        except Exception as e:
            logger.warning("Failed during stale sandbox cleanup: %s", e)

    def _container_name(self, user_id: str) -> str:
        return f"sandbox-{user_id[:12]}"

    @staticmethod
    def _is_read_only_error(error: Exception) -> bool:
        message = str(error).lower()
        return "read-only" in message or "readonly" in message

    def _has_writable_runtime_dirs(self, container) -> bool:
        """Verify tmpfs-backed runtime dirs are writable before reusing a container."""
        probe = (
            "from pathlib import Path\n"
            "for root in ('/tmp', '/code', '/home/sandbox', '/var/tmp'):\n"
            "    p = Path(root) / '.sandbox-write-test'\n"
            "    p.write_text('ok')\n"
            "    p.unlink()\n"
        )
        try:
            result = container.exec_run(
                cmd=["python", "-c", probe],
                demux=True,
            )
            return result.exit_code == 0
        except Exception as e:
            logger.warning("Sandbox writable-dir probe failed: %s", e)
            return False

    def _uses_current_image(self, container) -> bool:
        try:
            current_image = self.client.images.get(IMAGE)
            return container.image.id == current_image.id
        except Exception as e:
            logger.warning("Sandbox image freshness check failed: %s", e)
            return False

    def _is_reusable_container(self, container) -> bool:
        if not self._uses_current_image(container):
            return False
        return self._has_writable_runtime_dirs(container)

    def _remove_container(self, container):
        try:
            container.remove(force=True)
        except Exception as e:
            logger.warning("Failed to remove invalid sandbox container: %s", e)

    def get_or_create(self, user_id: str):
        if not self.client:
            raise RuntimeError("Docker not available")

        name = self._container_name(user_id)

        # Reuse existing running container
        if user_id in self._containers:
            container = self._containers[user_id]
            try:
                container.reload()
                if container.status == "running":
                    if self._is_reusable_container(container):
                        self._last_used[user_id] = time.time()
                        return container
                    logger.warning("Recreating sandbox %s because it is stale or not writable", name)
                    self._remove_container(container)
            except Exception:
                pass
            del self._containers[user_id]

        # Try to find existing container
        try:
            container = self.client.containers.get(name)
        except NotFound:
            container = None

        if container is not None:
            if container.status != "running":
                container.start()
            if self._is_reusable_container(container):
                self._containers[user_id] = container
                self._last_used[user_id] = time.time()
                return container
            logger.warning("Removing stale sandbox %s because it is stale or not writable", name)
            self._remove_container(container)

        # Create new container with security constraints
        # Use absolute path so Docker treats it as a bind mount, not a named volume
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        workspace = os.path.join(base_dir, "data", "workspaces", user_id)

        # Ensure host directories exist before bind-mounting
        os.makedirs(f"{workspace}/data", exist_ok=True)
        os.makedirs(f"{workspace}/output", exist_ok=True)

        container = self.client.containers.run(
            IMAGE,
            detach=True,
            stdin_open=True,
            name=name,
            labels=SANDBOX_LABELS,
            mem_limit="512m",
            memswap_limit="512m",
            nano_cpus=1_000_000_000,  # 1 CPU
            pids_limit=100,
            network_disabled=True,
            read_only=True,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            user="1000:1000",
            tmpfs={
                "/tmp": "rw,size=256m,mode=1777",
                "/home/sandbox": "rw,size=64m,uid=1000,gid=1000,mode=700",
                "/var/tmp": "rw,size=64m,mode=1777",
                "/code": "rw,size=64m,uid=1000,gid=1000,mode=700",
            },
            volumes={
                f"{workspace}/data": {"bind": "/data", "mode": "ro"},
                f"{workspace}/output": {"bind": "/output", "mode": "rw"},
            },
        )
        if not self._has_writable_runtime_dirs(container):
            self._remove_container(container)
            raise RuntimeError(
                "Sandbox container started without writable tmpfs directories. "
                "Remove existing sandbox-* containers and verify Docker supports tmpfs mounts."
            )
        self._containers[user_id] = container
        self._last_used[user_id] = time.time()
        return container

    async def _execute_in_container(self, user_id: str, code: str, timeout: int) -> dict:
        container = self.get_or_create(user_id)

        # Docker's put_archive cannot write to tmpfs paths when the container rootfs
        # is read-only, so stage the transient code file on the writable output bind.
        run_file = f".sandbox_run_{time.time_ns()}.py"
        run_path = f"/output/{run_file}"
        code_bytes = code.encode("utf-8")
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=run_file)
            info.size = len(code_bytes)
            info.mode = 0o600
            info.uid = 1000
            info.gid = 1000
            info.uname = "sandbox"
            info.gname = "sandbox"
            tar.addfile(info, io.BytesIO(code_bytes))
        tar_stream.seek(0)
        await asyncio.to_thread(container.put_archive, "/output", tar_stream)

        # Execute through executor.py — enforces import/builtin/path/timeout/output limits
        effective_timeout = max(1, min(timeout, 300))
        try:
            exit_code, (stdout, stderr) = await asyncio.wait_for(
                asyncio.to_thread(
                    container.exec_run,
                    cmd=["python", "-u", "/executor.py", run_path],
                    demux=True,
                    environment={"TIMEOUT_SECONDS": str(effective_timeout)},
                ),
                timeout=effective_timeout + 5,
            )
        except asyncio.TimeoutError:
            # Destroy the container to avoid reusing a potentially stuck container
            self.destroy(user_id)
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Error: Execution timed out ({effective_timeout}s)",
            }
        except Exception as e:
            logger.error("Sandbox execution error: %s", e)
            raise RuntimeError(f"Sandbox execution failed: {e}")

        self._last_used[user_id] = time.time()

        # Best-effort cleanup of temp file
        try:
            await asyncio.to_thread(
                container.exec_run,
                cmd=["rm", "-f", run_path],
                demux=True,
            )
        except Exception:
            pass

        out = (stdout or b"").decode("utf-8", errors="replace")
        err = (stderr or b"").decode("utf-8", errors="replace")
        if exit_code != 0 and err:
            raise RuntimeError(f"Code execution failed:\n{err}")
        return {"exit_code": exit_code, "stdout": out, "stderr": err}

    async def execute(self, user_id: str, code: str, timeout: int = 60) -> dict:
        """Execute code in user's sandbox container via executor.py (with full restrictions)."""
        if not code or not code.strip():
            return {"exit_code": 0, "stdout": "", "stderr": ""}
        if len(code.encode("utf-8")) > 50 * 1024:
            raise ValueError("Code too large (max 50KB)")

        try:
            return await self._execute_in_container(user_id, code, timeout)
        except Exception as e:
            if not self._is_read_only_error(e):
                raise
            logger.warning("Recreating sandbox after read-only filesystem error: %s", e)
            self.destroy(user_id)
            return await self._execute_in_container(user_id, code, timeout)

    def cleanup_idle(self):
        now = time.time()
        for user_id in list(self._last_used.keys()):
            if now - self._last_used[user_id] > IDLE_TIMEOUT:
                self.destroy(user_id)

    def destroy(self, user_id: str):
        if user_id in self._containers:
            try:
                self._containers[user_id].remove(force=True)
            except Exception:
                pass
            del self._containers[user_id]
            self._last_used.pop(user_id, None)

    def shutdown(self):
        for user_id in list(self._containers.keys()):
            self.destroy(user_id)


# Global singleton
sandbox_manager = DockerSandboxManager()
