"""
时钟类 syscall。
"""

from datetime import datetime
from typing import Dict

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult


def sys_clock(args: Dict[str, str]) -> SyscallResult:
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
        return SyscallResult(content=now.isoformat())
    if fmt == "date":
        return SyscallResult(content=now.strftime("%Y-%m-%d"))
    if fmt == "time":
        return SyscallResult(content=now.strftime("%H:%M:%S"))

    utc_offset_hours = now.utcoffset().total_seconds() / 3600
    offset_int = int(utc_offset_hours)
    return SyscallResult(content=f"{now.strftime('%Y-%m-%d %H:%M:%S')} (UTC{offset_int:+d})")
