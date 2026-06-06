"""
文件读写类 syscall 与路径安全工具。
"""

from pathlib import Path
from typing import Dict

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult
from hivememory.core.mtp.exceptions import (
    SyscallExecutionError,
    SyscallInvalidArgumentError,
    SyscallPermissionDeniedError,
)
from hivememory.i18n.syscall_runtime import get_syscall_info_text


def _resolve_safe_path(path_str: str, workspace: str) -> Path:
    """
    解析并验证路径安全性 (防止路径穿越)。
    """
    workspace_resolved = Path(workspace).resolve()
    target = (workspace_resolved / path_str).resolve()

    if not target.is_relative_to(workspace_resolved):
        raise SyscallPermissionDeniedError(
            message_key="syscall.file_read.path_denied",
            params={"path": path_str},
        )

    return target


def sys_read_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> SyscallResult:
    """
    读取工作区文件 (Chapter 8.1)。
    """
    path_str = args.get("path", "")
    if not path_str:
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_read.missing_path",
            params={"arg": "path"},
        )

    target = _resolve_safe_path(path_str, workspace)

    if not target.exists():
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_read.not_found",
            params={"path": path_str},
        )

    if not target.is_file():
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_read.not_file",
            params={"path": path_str},
        )

    try:
        with open(target, "rb") as f:
            head = f.read(512)
        if b"\x00" in head:
            raise SyscallInvalidArgumentError(
                message_key="syscall.file_read.binary_file",
                params={"path": path_str},
            )
    except OSError as exc:
        raise SyscallExecutionError(
            message_key="syscall.file_read.read_failed",
            params={"path": path_str, "detail": "The file may be locked or inaccessible."},
            cause=exc,
        ) from exc

    file_size = target.stat().st_size
    truncated = file_size > max_bytes

    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read(max_bytes)
    except UnicodeDecodeError:
        try:
            with open(target, "r", encoding="latin-1") as f:
                content = f.read(max_bytes)
        except OSError as exc:
            raise SyscallExecutionError(
                message_key="syscall.file_read.read_failed",
                params={"path": path_str, "detail": "The file may be locked or inaccessible."},
                cause=exc,
            ) from exc
    except OSError as exc:
        raise SyscallExecutionError(
            message_key="syscall.file_read.read_failed",
            params={"path": path_str, "detail": "The file may be locked or inaccessible."},
            cause=exc,
        ) from exc

    result = f"<content>\n{content}\n</content>"
    if truncated:
        result += "\n" + get_syscall_info_text(
            "syscall.file_read.truncated",
            {"max_bytes": max_bytes, "file_size": file_size},
        )
    return SyscallResult(content=result)


def sys_write_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> SyscallResult:
    """
    写入工作区文件 (Chapter 8.1)。
    """
    path_str = args.get("path", "")
    if not path_str:
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_write.missing_path",
            params={"arg": "path"},
        )

    content = args.get("content", "")
    if not content:
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_write.missing_content",
            params={"arg": "content"},
        )

    mode = args.get("mode", "overwrite")
    if mode not in ("overwrite", "append"):
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_write.invalid_mode",
            params={"mode": mode},
        )

    content_bytes = content.encode("utf-8")
    if len(content_bytes) > max_bytes:
        raise SyscallInvalidArgumentError(
            message_key="syscall.file_write.content_too_large",
            params={"size": len(content_bytes), "max_bytes": max_bytes},
        )

    try:
        target = _resolve_safe_path(path_str, workspace)
    except SyscallPermissionDeniedError as exc:
        raise SyscallPermissionDeniedError(
            message_key="syscall.file_write.path_denied",
            params={"path": path_str},
            cause=exc,
        ) from exc

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(target, write_mode, encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        raise SyscallExecutionError(
            message_key="syscall.file_write.write_failed",
            params={"path": path_str, "detail": "The path may be read-only or inaccessible."},
            cause=exc,
        ) from exc

    return SyscallResult(
        content=get_syscall_info_text(
            "syscall.file_write.success",
            {"name": target.name, "bytes": len(content_bytes)},
        )
    )
