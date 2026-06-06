"""
Python REPL 类 syscall 与沙箱执行实现。
"""

import builtins
import contextlib
import io
import threading
from typing import Any, Dict, Optional

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult


# 安全 builtins 白名单 (Section 4.3.1 MVP)
# 禁止: import, open, exec, eval, compile, __import__, globals, locals, vars,
#        getattr, setattr, delattr, breakpoint, exit, quit, input
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


def _blocked_import(*args, **kwargs):
    """阻止 import 语句。"""
    raise ImportError(
        "import is not allowed in the restricted REPL. "
        "Only built-in functions are available."
    )


class _TimeoutError(Exception):
    """REPL 执行超时。"""


def _run_code_in_thread(
    code: str,
    namespace: dict,
    stdout_capture: io.StringIO,
    timeout_seconds: int,
) -> None:
    """
    在子线程中执行代码，主线程等待超时 (跨平台)。
    """
    result_holder = {"error": None}

    def target():
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<mtp_repl>", "exec"), namespace)
        except Exception as e:  # pragma: no cover - 异常信息已转为返回值
            result_holder["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise _TimeoutError(f"Execution timed out after {timeout_seconds}s.")

    if result_holder["error"] is not None:
        raise result_holder["error"]


def execute_sandboxed(
    code: str,
    *,
    namespace_extras: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 10,
) -> SyscallResult:
    """
    在受限沙箱中执行 Python 代码。
    """
    restricted_builtins = {
        k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)
    }
    restricted_builtins["__import__"] = _blocked_import

    namespace: dict = {"__builtins__": restricted_builtins}
    if namespace_extras:
        namespace.update(namespace_extras)

    stdout_capture = io.StringIO()

    try:
        _run_code_in_thread(
            code=code,
            namespace=namespace,
            stdout_capture=stdout_capture,
            timeout_seconds=timeout_seconds,
        )
    except _TimeoutError:
        return SyscallResult(
            ok=False,
            content=f"Execution timed out after {timeout_seconds}s.",
            error_code="mtp.system.tool_error",
        )
    except ImportError as e:
        return SyscallResult(ok=False, content=str(e), error_code="mtp.system.tool_error")
    except Exception:
        return SyscallResult(
            ok=False,
            content="Python execution failed. Check your code for runtime errors.",
            error_code="mtp.system.tool_error",
        )

    output = stdout_capture.getvalue().strip()
    return SyscallResult(ok=True, content=f"Stdout: {output}" if output else "Executed successfully (no output).")


def sys_python_repl(args: Dict[str, str], *, timeout_seconds: int = 10) -> SyscallResult:
    """
    受限 Python REPL (Section 4.3.1 MVP / Chapter 8.3)。
    """
    code = args.get("code", "")
    if not code:
        return SyscallResult(ok=False, content="'code' argument is required.", error_code="mtp.argument.invalid")

    return execute_sandboxed(code, timeout_seconds=timeout_seconds)
