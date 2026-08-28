"""Runs inside the Docker sandbox. Reads Python code from stdin or a file, executes it safely."""

import sys
import os
import builtins
import threading
import io
import traceback

BLOCKED_MODULES = {"os", "subprocess", "shutil", "socket", "http", "ctypes", "signal", "importlib"}
BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "open"}
ALLOWED_PATHS = ("/data/", "/output/")
MAX_OUTPUT_BYTES = 1 * 1024 * 1024  # 1MB
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", 60))
_original_open = builtins.open  # captured at import time
_original_exec = builtins.exec  # keep executor internals working after builtins are restricted


def restricted_open(path, *args, **kwargs):
    path_str = os.fspath(path)
    if not any(path_str.startswith(p) for p in ALLOWED_PATHS):
        raise PermissionError(f"open() denied: {path_str} not in allowed paths")
    return _original_open(path, *args, **kwargs)


def check_imports(code: str):
    import ast
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    raise ImportError(f"Module '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_MODULES:
                raise ImportError(f"Module '{node.module}' is not allowed")


def main():
    # Accept code from a file path argument (preferred) or stdin (fallback)
    if len(sys.argv) > 1:
        code_path = sys.argv[1]
        try:
            with open(code_path) as f:
                code = f.read()
        except FileNotFoundError:
            print(f"Error: code file not found: {code_path}", file=sys.stderr)
            sys.exit(1)
    else:
        code = sys.stdin.read()

    if len(code) > 50 * 1024:
        print("Error: code too large (max 50KB)", file=sys.stderr)
        sys.exit(1)

    check_imports(code)

    # Use threading-based timeout (works on all platforms including Windows)
    result = {}
    def run_code():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        sys.stdout = captured_out
        sys.stderr = captured_err

        builtins.open = restricted_open
        for name in BLOCKED_BUILTINS:
            if hasattr(builtins, name):
                delattr(builtins, name)

        try:
            _original_exec(code, {"__builtins__": builtins})
        except Exception:
            traceback.print_exc(file=captured_err)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result["out"] = captured_out.getvalue()
            result["err"] = captured_err.getvalue()

    thread = threading.Thread(target=run_code, daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT_SECONDS)
    if thread.is_alive():
        print(f"Error: Execution timed out ({TIMEOUT_SECONDS}s)", file=sys.stderr)
        sys.exit(1)

    out = result.get("out", "")
    err = result.get("err", "")

    if len(out) > MAX_OUTPUT_BYTES:
        out = out[:MAX_OUTPUT_BYTES] + "\n... [output truncated]"
    if len(err) > MAX_OUTPUT_BYTES:
        err = err[:MAX_OUTPUT_BYTES] + "\n... [error truncated]"

    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)


if __name__ == "__main__":
    main()
