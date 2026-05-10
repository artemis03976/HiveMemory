"""
文件读写类 syscall 与路径安全工具。
"""

from pathlib import Path
from typing import Dict


def _resolve_safe_path(path_str: str, workspace: str) -> Path:
    """
    解析并验证路径安全性 (防止路径穿越)。
    """
    workspace_resolved = Path(workspace).resolve()
    target = (workspace_resolved / path_str).resolve()

    if not target.is_relative_to(workspace_resolved):
        raise PermissionError(
            f"Access denied: path '{path_str}' escapes workspace boundary."
        )

    return target


def sys_read_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> str:
    """
    读取工作区文件 (Chapter 8.1)。
    """
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required."

    try:
        target = _resolve_safe_path(path_str, workspace)
    except PermissionError as e:
        return f"Error: {e}"

    if not target.exists():
        return f"Error: File not found: '{path_str}'"

    if not target.is_file():
        return f"Error: '{path_str}' is not a file."

    try:
        with open(target, "rb") as f:
            head = f.read(512)
        if b"\x00" in head:
            return f"Error: '{path_str}' appears to be a binary file."
    except OSError:
        return "Error: Cannot read file. The file may be locked or inaccessible."

    file_size = target.stat().st_size
    truncated = file_size > max_bytes

    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read(max_bytes)
    except UnicodeDecodeError:
        try:
            with open(target, "r", encoding="latin-1") as f:
                content = f.read(max_bytes)
        except OSError:
            return "Error: Cannot read file. The file may be locked or inaccessible."
    except OSError:
        return "Error: Cannot read file. The file may be locked or inaccessible."

    result = f"<content>\n{content}\n</content>"
    if truncated:
        result += f"\n[Truncated: showing first {max_bytes} bytes of {file_size} bytes]"
    return result


def sys_write_file(
    args: Dict[str, str],
    *,
    workspace: str = "./workspace",
    max_bytes: int = 102400,
) -> str:
    """
    写入工作区文件 (Chapter 8.1)。
    """
    path_str = args.get("path", "")
    if not path_str:
        return "Error: 'path' argument is required."

    content = args.get("content", "")
    if not content:
        return "Error: 'content' argument is required."

    mode = args.get("mode", "overwrite")
    if mode not in ("overwrite", "append"):
        return f"Error: Invalid mode '{mode}'. Use 'overwrite' or 'append'."

    content_bytes = content.encode("utf-8")
    if len(content_bytes) > max_bytes:
        return (
            f"Error: Content too large ({len(content_bytes)} bytes). "
            f"Maximum allowed: {max_bytes} bytes."
        )

    try:
        target = _resolve_safe_path(path_str, workspace)
    except PermissionError as e:
        return f"Error: {e}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(target, write_mode, encoding="utf-8") as f:
            f.write(content)
    except OSError:
        return "Error: Cannot write file. The path may be read-only or inaccessible."

    return f"Success: File '{target.name}' saved ({len(content_bytes)} bytes)."

