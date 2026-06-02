"""
Alice Runtime Level-0 syscalls 模块。
"""

from hivememory.agent_runtime.syscalls.clock import sys_clock
from hivememory.agent_runtime.syscalls.file_io import sys_read_file, sys_write_file
from hivememory.agent_runtime.syscalls.registry import build_kernel_registry
from hivememory.agent_runtime.syscalls.repl import execute_sandboxed, sys_python_repl
from hivememory.agent_runtime.syscalls.types import KernelSyscall
from hivememory.agent_runtime.syscalls.web_search import sys_web_search

__all__ = [
    "KernelSyscall",
    "sys_clock",
    "execute_sandboxed",
    "sys_python_repl",
    "sys_web_search",
    "sys_read_file",
    "sys_write_file",
    "build_kernel_registry",
]
