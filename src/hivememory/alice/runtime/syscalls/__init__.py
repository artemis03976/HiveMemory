"""
Alice Runtime Level-0 syscalls 模块。
"""

from hivememory.alice.runtime.syscalls.clock import sys_clock
from hivememory.alice.runtime.syscalls.file_io import sys_read_file, sys_write_file
from hivememory.alice.runtime.syscalls.registry import build_kernel_registry
from hivememory.alice.runtime.syscalls.repl import execute_sandboxed, sys_python_repl
from hivememory.alice.runtime.syscalls.types import KernelSyscall
from hivememory.alice.runtime.syscalls.web_search import sys_web_search

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
