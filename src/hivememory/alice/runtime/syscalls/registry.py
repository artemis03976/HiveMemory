"""
syscall 注册表构建器。
"""

from functools import partial
from typing import Dict

from hivememory.alice.runtime.syscalls.clock import sys_clock
from hivememory.alice.runtime.syscalls.file_io import sys_read_file, sys_write_file
from hivememory.alice.runtime.syscalls.repl import sys_python_repl
from hivememory.alice.runtime.syscalls.types import KernelSyscall
from hivememory.alice.runtime.syscalls.web_search import sys_web_search


def build_kernel_registry(
    *,
    python_repl_timeout: int = 10,
    workspace_path: str = "./workspace",
    file_read_max_bytes: int = 102400,
    file_write_max_bytes: int = 102400,
    web_search_timeout: int = 15,
) -> Dict[str, KernelSyscall]:
    """
    构建内核工具注册表 (KERNEL_REGISTRY)。
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
