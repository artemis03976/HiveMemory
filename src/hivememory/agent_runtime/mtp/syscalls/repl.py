"""
Python REPL 类 syscall 与沙箱执行实现。
"""

import builtins
import json
import subprocess
import sys
from typing import Any, Dict, Optional

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult
from hivememory.core.mtp.exceptions import (
    SyscallExecutionError,
    SyscallInvalidArgumentError,
    SyscallPermissionDeniedError,
    SyscallTimeoutError,
)
from hivememory.i18n.syscall_runtime import get_syscall_info_text


# 安全 builtins 白名单。禁止 import/open/exec/eval/compile 等能力。
_SAFE_BUILTINS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytes",
        "bytearray",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)


class _TimeoutError(Exception):
    """REPL 执行超时。"""


def _run_code_in_process(
    code: str,
    namespace_extras: dict[str, Any],
    timeout_seconds: int,
) -> str:
    """
    在子进程中执行代码，超时后终止进程。

    Windows 上线程无法安全停止 `while True: pass` 这类忙循环，因此 timeout
    路径使用 subprocess 隔离，避免 runtime 被遗留执行单元拖住。
    """
    payload = {
        "code": code,
        "namespace_extras": namespace_extras,
        "safe_builtins": sorted(_SAFE_BUILTINS),
    }
    runner = r"""
import builtins
import contextlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())

def blocked_import(*args, **kwargs):
    raise ImportError("import is not allowed in the restricted REPL.")

restricted = {
    key: getattr(builtins, key)
    for key in payload["safe_builtins"]
    if hasattr(builtins, key)
}
restricted["__import__"] = blocked_import
namespace = {"__builtins__": restricted}
namespace.update(payload.get("namespace_extras") or {})

stdout_capture = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout_capture):
        exec(compile(payload["code"], "<mtp_repl>", "exec"), namespace)
except ImportError as exc:
    print(json.dumps({"status": "import_error", "message": str(exc)}))
except BaseException as exc:
    print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}))
else:
    print(json.dumps({"status": "ok", "stdout": stdout_capture.getvalue()}))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _TimeoutError(f"Execution timed out after {timeout_seconds}s.") from exc

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Python execution failed.")

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Python execution failed without structured result.") from exc

    status = result.get("status")
    if status == "import_error":
        raise ImportError(result.get("message") or "import is not allowed.")
    if status == "error":
        raise RuntimeError(result.get("message") or "Python execution failed.")
    return result.get("stdout", "")


def execute_sandboxed(
    code: str,
    *,
    namespace_extras: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 10,
) -> SyscallResult:
    """Execute Python code in a restricted subprocess sandbox."""
    try:
        output = _run_code_in_process(
            code=code,
            namespace_extras=dict(namespace_extras or {}),
            timeout_seconds=timeout_seconds,
        )
    except _TimeoutError as exc:
        raise SyscallTimeoutError(
            message_key="syscall.repl.timeout",
            params={"timeout_seconds": timeout_seconds},
            cause=exc,
        ) from exc
    except ImportError as exc:
        raise SyscallPermissionDeniedError(
            message_key="syscall.repl.import_blocked",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise SyscallExecutionError(
            message_key="syscall.repl.execution_failed",
            params={"detail": "Check your code for runtime errors."},
            cause=exc,
        ) from exc

    output = output.strip()
    if output:
        return SyscallResult(
            content=get_syscall_info_text("syscall.repl.stdout", {"output": output})
        )
    return SyscallResult(content=get_syscall_info_text("syscall.repl.no_output"))


def sys_python_repl(args: Dict[str, str], *, timeout_seconds: int = 10) -> SyscallResult:
    """Restricted Python REPL syscall."""
    code = args.get("code", "")
    if not code:
        raise SyscallInvalidArgumentError(
            message_key="syscall.repl.missing_code",
            params={"arg": "code"},
        )

    return execute_sandboxed(code, timeout_seconds=timeout_seconds)
