"""
Kernel Syscalls (内核级指令)

Level 0 硬编码工具集，随系统启动加载，Zero Latency。
通过 KERNEL_REGISTRY 字典实现快速路径分发 (Section 4.2.1)。

MVP 实现:
- sys_clock: 获取当前系统时间
- sys_python_repl: 受限 Python REPL (Section 4.3.1)
- sys_web_search: 网络搜索 (Section 8.2)
- sys_read_file: 读取工作区文件 (Section 8.1)
- sys_write_file: 写入工作区文件 (Section 8.1)

对应设计文档: MemoryToolProtocol.md Chapter 4 & 8

作者: HiveMemory Team
版本: 1.0
"""

import builtins
import contextlib
import io
import logging
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict

logger = logging.getLogger(__name__)


# ========== 类型定义 ==========

@dataclass(frozen=True)
class KernelSyscall:
    """
    内核级工具定义 (Level 0)

    每个 syscall 包含一个 handler 函数和描述。
    handler 签名: (args: Dict[str, str]) -> str
    """
    handler: Callable[[Dict[str, str]], str]
    description: str


# ========== sys_clock ==========

def sys_clock(args: Dict[str, str]) -> str:
    """
    获取当前系统时间 (Chapter 8.4)

    Args (MTP):
        format: 输出格式 (可选)
            - "default": "YYYY-MM-DD HH:MM:SS (UTC+X)"
            - "iso": ISO 8601 格式
            - "date": "YYYY-MM-DD"
            - "time": "HH:MM:SS"
    """
    fmt = args.get("format", "default")
    now = datetime.now().astimezone()

    if fmt == "iso":
        return now.isoformat()
    elif fmt == "date":
        return now.strftime("%Y-%m-%d")
    elif fmt == "time":
        return now.strftime("%H:%M:%S")
    else:
        utc_offset_hours = now.utcoffset().total_seconds() / 3600
        offset_int = int(utc_offset_hours)
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')} (UTC{offset_int:+d})"


# ========== sys_python_repl ==========

# 安全 builtins 白名单 (Section 4.3.1 MVP)
# 禁止: import, open, exec, eval, compile, __import__, globals, locals, vars,
#        getattr, setattr, delattr, breakpoint, exit, quit, input
_SAFE_BUILTINS = frozenset({
    "abs", "all", "any", "bin", "bool", "bytes", "bytearray",
    "callable", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next",
    "oct", "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
})


def _blocked_import(*args, **kwargs):
    """阻止 import 语句"""
    raise ImportError(
        "import is not allowed in the restricted REPL. "
        "Only built-in functions are available."
    )


class _TimeoutError(Exception):
    """REPL 执行超时"""
    pass


def _run_code_in_thread(
    code: str,
    namespace: dict,
    stdout_capture: io.StringIO,
    timeout_seconds: int,
) -> str:
    """
    在子线程中执行代码，主线程等待超时 (跨平台)

    Args:
        code: 要执行的代码
        namespace: 执行命名空间
        stdout_capture: stdout 捕获器
        timeout_seconds: 超时秒数

    Returns:
        str: 执行结果或错误信息

    Raises:
        _TimeoutError: 执行超时
    """
    result_holder = {"error": None}

    def target():
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<mtp_repl>", "exec"), namespace)
        except Exception as e:
            result_holder["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # 线程仍在运行 — 超时
        # daemon=True 确保不会阻止进程退出
        raise _TimeoutError(
            f"Execution timed out after {timeout_seconds}s."
        )

    if result_holder["error"] is not None:
        raise result_holder["error"]


def sys_python_repl(args: Dict[str, str], *, timeout_seconds: int = 10) -> str:
    """
    受限 Python REPL (Section 4.3.1 MVP / Chapter 8.3)

    在受限沙箱中执行 Python 代码。禁止 import、文件操作等危险操作。

    Args (MTP):
        code: 要执行的 Python 代码 (必需)

    安全模型:
        - 白名单 builtins (禁止 import, open, exec, eval 等)
        - 子线程执行 + 超时熔断
        - 隔离命名空间
    """
    code = args.get("code", "")
    if not code:
        return "Error: 'code' argument is required."

    # 构建受限 builtins
    restricted_builtins = {
        k: getattr(builtins, k)
        for k in _SAFE_BUILTINS
        if hasattr(builtins, k)
    }
    restricted_builtins["__import__"] = _blocked_import

    # 隔离命名空间
    namespace = {"__builtins__": restricted_builtins}
    stdout_capture = io.StringIO()

    try:
        _run_code_in_thread(
            code=code,
            namespace=namespace,
            stdout_capture=stdout_capture,
            timeout_seconds=timeout_seconds,
        )
    except _TimeoutError:
        return f"Error: Execution timed out after {timeout_seconds}s."
    except Exception:
        # 返回最后 3 行 traceback (Section 4.4)
        tb_lines = traceback.format_exc().strip().split("\n")
        return "Error:\n" + "\n".join(tb_lines[-3:])

    output = stdout_capture.getvalue().strip()
    return f"Stdout: {output}" if output else "Executed successfully (no output)."


# ========== sys_web_search ==========

def sys_web_search(args: Dict[str, str], *, timeout_seconds: int = 15) -> str:
    """
    网络搜索 (Chapter 8.2)

    调用 DuckDuckGo 搜索引擎，返回格式化的搜索结果。

    Args (MTP):
        query: 搜索关键词 (必需)
        num: 返回结果数量 (可选, 默认 3, 范围 [1, 10])

    依赖: duckduckgo-search (可选安装: pip install hivememory[search])
    """
    query = args.get("query", "")
    if not query:
        return "Error: 'query' argument is required."

    num_str = args.get("num", "3")
    try:
        num = max(1, min(10, int(num_str)))
    except (ValueError, TypeError):
        num = 3

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return (
            "Error: duckduckgo-search package is not installed. "
            "Install it with: pip install hivememory[search]"
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num))
    except Exception as e:
        return f"Error: Web search failed: {e}"

    if not results:
        return f"No results found for query: '{query}'"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "N/A")
        snippet = r.get("body", "N/A")
        url = r.get("href", "N/A")
        lines.append(f"[{i}] Title: {title}\nSnippet: {snippet}\nURL: {url}")

    return "\n\n".join(lines)


# ========== 路径安全工具 ==========

def _resolve_safe_path(path_str: str, workspace: str) -> Path:
    """
    解析并验证路径安全性 (防止路径穿越)

    Args:
        path_str: 用户提供的相对路径
        workspace: 工作区根目录

    Returns:
        Path: 解析后的绝对路径

    Raises:
        PermissionError: 路径逃逸出工作区
    """
    workspace_resolved = Path(workspace).resolve()
    target = (workspace_resolved / path_str).resolve()

    if not target.is_relative_to(workspace_resolved):
        raise PermissionError(
            f"Access denied: path '{path_str}' escapes workspace boundary."
        )

    return target


# ========== sys_read_file ==========

def sys_read_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> str:
    """
    读取工作区文件 (Chapter 8.1)

    Args (MTP):
        path: 文件相对路径 (必需, 相对于 workspace)

    安全模型:
        - 路径穿越防护: resolve() + is_relative_to()
        - 二进制文件检测: 前 512 字节含 null byte 则拒绝
        - 大小限制: 超过 max_bytes 截断
    """
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required."

    try:
        target = _resolve_safe_path(path_str, workspace)
    except PermissionError as e:
        return f"Error: {e}"

    if not target.exists():
        return f"Error: File not found: '{path_str}'"

    if not target.is_file():
        return f"Error: '{path_str}' is not a file."

    # 二进制文件检测
    try:
        with open(target, "rb") as f:
            head = f.read(512)
        if b"\x00" in head:
            return f"Error: '{path_str}' appears to be a binary file."
    except OSError as e:
        return f"Error: Cannot read file: {e}"

    # 读取文本内容
    file_size = target.stat().st_size
    truncated = file_size > max_bytes

    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read(max_bytes)
    except UnicodeDecodeError:
        try:
            with open(target, "r", encoding="latin-1") as f:
                content = f.read(max_bytes)
        except OSError as e:
            return f"Error: Cannot read file: {e}"
    except OSError as e:
        return f"Error: Cannot read file: {e}"

    result = f"<content>\n{content}\n</content>"
    if truncated:
        result += f"\n[Truncated: showing first {max_bytes} bytes of {file_size} bytes]"
    return result


# ========== sys_write_file ==========

def sys_write_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> str:
    """
    写入工作区文件 (Chapter 8.1)

    Args (MTP):
        path: 文件相对路径 (必需, 相对于 workspace)
        content: 要写入的内容 (必需)
        mode: 写入模式 (可选, "overwrite" 或 "append", 默认 "overwrite")

    安全模型:
        - 路径穿越防护: resolve() + is_relative_to()
        - 大小限制: content 编码后超过 max_bytes 拒绝
        - 自动创建父目录
    """
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required."

    content = args.get("content", "")
    if not content:
        return "Error: 'content' argument is required."

    mode = args.get("mode", "overwrite")
    if mode not in ("overwrite", "append"):
        return f"Error: Invalid mode '{mode}'. Use 'overwrite' or 'append'."

    content_bytes = content.encode("utf-8")
    if len(content_bytes) > max_bytes:
        return (
            f"Error: Content too large ({len(content_bytes)} bytes). "
            f"Maximum allowed: {max_bytes} bytes."
        )

    try:
        target = _resolve_safe_path(path_str, workspace)
    except PermissionError as e:
        return f"Error: {e}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(target, write_mode, encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: Cannot write file: {e}"

    return f"Success: File '{target.name}' saved ({len(content_bytes)} bytes)."


# ========== 注册表构建 ==========

def build_kernel_registry(
    *,
    python_repl_timeout: int = 10,
    workspace_path: str = "./workspace",
    file_read_max_bytes: int = 102400,
    file_write_max_bytes: int = 102400,
    web_search_timeout: int = 15,
) -> Dict[str, KernelSyscall]:
    """
    构建内核工具注册表 (KERNEL_REGISTRY)

    返回 Dict[str, KernelSyscall]，供 Koakuma 快速路径分发使用。

    Args:
        python_repl_timeout: sys_python_repl 超时秒数
        workspace_path: 工作区根目录路径
        file_read_max_bytes: sys_read_file 最大读取字节数
        file_write_max_bytes: sys_write_file 最大写入字节数
        web_search_timeout: sys_web_search 超时秒数

    Returns:
        Dict[str, KernelSyscall]: 注册表
    """
    return {
        "sys_clock": KernelSyscall(
            handler=sys_clock,
            description="Get current date, time, and timezone.",
        ),
        "sys_python_repl": KernelSyscall(
            handler=partial(
                sys_python_repl,
                timeout_seconds=python_repl_timeout,
            ),
            description="Execute Python code for calculation or data processing.",
        ),
        "sys_web_search": KernelSyscall(
            handler=partial(
                sys_web_search,
                timeout_seconds=web_search_timeout,
            ),
            description="Search the internet for latest information.",
        ),
        "sys_read_file": KernelSyscall(
            handler=partial(
                sys_read_file,
                workspace=workspace_path,
                max_bytes=file_read_max_bytes,
            ),
            description="Read a file from the workspace directory.",
        ),
        "sys_write_file": KernelSyscall(
            handler=partial(
                sys_write_file,
                workspace=workspace_path,
                max_bytes=file_write_max_bytes,
            ),
            description="Write content to a file in the workspace directory.",
        ),
    }


__all__ = [
    "KernelSyscall",
    "sys_clock",
    "sys_python_repl",
    "sys_web_search",
    "sys_read_file",
    "sys_write_file",
    "build_kernel_registry",
]
