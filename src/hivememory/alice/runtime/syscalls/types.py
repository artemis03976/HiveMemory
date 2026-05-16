"""
syscalls 公共类型定义。
"""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class KernelSyscall:
    """
    内核级工具定义 (Level 0)

    每个 syscall 包含一个 handler 函数和描述。
    handler 签名: (args: Dict[str, str]) -> str
    """

    handler: Callable[[Dict[str, str]], str]
    description: str
