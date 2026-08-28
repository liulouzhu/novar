"""Python 代码执行工具 - Docker 沙箱隔离执行"""

import asyncio
import logging

from tools.result import ok, fail, truncate_pair

logger = logging.getLogger("research-server.code_execute")


async def handle_code_execute(args: dict, user_id: str = None) -> str:
    """Execute Python code in a Docker sandbox. 返回统一 JSON。"""
    code = args.get("code", "").strip()
    if not code:
        return fail("code_execute", "empty code")

    timeout = min(args.get("timeout", 60), 300)

    if not user_id:
        return fail("code_execute", "user context required for code execution")

    try:
        from web.backend.sandbox.manager import sandbox_manager
        result = await sandbox_manager.execute(user_id, code, timeout)

        stdout, stderr = truncate_pair(result.get("stdout", ""), result.get("stderr", ""))
        exit_code = result.get("exit_code", 0)

        data = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }

        if exit_code != 0:
            return fail("code_execute", f"代码执行失败 (exit_code={exit_code})", data=data)

        return ok(
            "code_execute",
            data,
            summary="代码执行完成 (exit_code=0)",
        )
    except Exception as e:
        return fail("code_execute", f"沙箱执行异常: {e}")
