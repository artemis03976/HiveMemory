"""
syscalls 公共类型定义。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class SyscallResult:
    """
    syscall 执行结果。

    替代裸字符串返回值，消除 startswith("Error") 字符串嗅探。
    error_code 对应 MTPErrorInfo.code 命名空间，i18n 化时使用。
    """
    ok: bool
    content: str
    error_code: Optional[str] = field(default=None)


@dataclass(frozen=True)
class KernelSyscall:
    """
    内核级工具定义 (Level 0)

    handler 签名: (args: Dict[str, str]) -> SyscallResult
    """

    handler: Callable[..., SyscallResult]
    description: str
