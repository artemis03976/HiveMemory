"""
Alice Runtime Level-0 syscalls 模块。
"""

from hivememory.agent_runtime.mtp.syscalls.clock import sys_clock
from hivememory.agent_runtime.mtp.syscalls.file_io import sys_read_file, sys_write_file
from hivememory.agent_runtime.mtp.syscalls.registry import build_kernel_registry
from hivememory.agent_runtime.mtp.syscalls.repl import execute_sandboxed, sys_python_repl
from hivememory.agent_runtime.mtp.syscalls.types import KernelSyscall, SyscallResult
from hivememory.agent_runtime.mtp.syscalls.web_search import sys_web_search

__all__ = [
    "KernelSyscall",
    "SyscallResult",
    "sys_clock",
    "execute_sandboxed",
    "sys_python_repl",
    "sys_web_search",
    "sys_read_file",
    "sys_write_file",
    "build_kernel_registry",
]
