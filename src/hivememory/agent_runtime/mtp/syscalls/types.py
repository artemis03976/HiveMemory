"""syscall 公共类型定义。"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class SyscallResult:
    """
    syscall 成功执行结果。

    失败路径由 syscall handler 直接抛结构化 MTPError。
    """

    content: str


@dataclass(frozen=True)
class KernelSyscall:
    """
    内核级工具定义（Level 0）。

    handler 签名: (args: Dict[str, str]) -> SyscallResult
    """

    handler: Callable[..., SyscallResult]
    description: str
