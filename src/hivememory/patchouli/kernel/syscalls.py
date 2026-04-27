"""
Kernel syscalls 兼容层。

新的实现位于 `hivememory.patchouli.tools.syscalls`。
当前模块保留以兼容历史 import 路径。
"""

from hivememory.patchouli.tools.syscalls import (
    KernelSyscall,
    build_kernel_registry,
    execute_sandboxed,
    sys_clock,
    sys_python_repl,
    sys_read_file,
    sys_web_search,
    sys_write_file,
)

__all__ = [
    "KernelSyscall",
    "execute_sandboxed",
    "sys_clock",
    "sys_python_repl",
    "sys_web_search",
    "sys_read_file",
    "sys_write_file",
    "build_kernel_registry",
]
