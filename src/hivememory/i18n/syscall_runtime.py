"""Syscall runtime i18n templates."""

from __future__ import annotations

from typing import Any

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language


_SYSCALL_ERROR_TEXT_ZH: dict[str, str] = {
    "syscall.file_read.missing_path": (
        "[Invalid Argument] file_read 缺少 \"{arg}\" 参数。\n"
        "Suggestion: 请提供要读取的工作区相对路径。"
    ),
    "syscall.file_read.path_denied": (
        "[Permission Denied] 路径 '{path}' 超出工作区边界。\n"
        "Suggestion: 请使用工作区内的相对路径。"
    ),
    "syscall.file_read.not_found": (
        "[Invalid Argument] 文件未找到：'{path}'。\n"
        "Suggestion: 请检查路径，或先搜索/列出可用文件。"
    ),
    "syscall.file_read.not_file": (
        "[Invalid Argument] '{path}' 不是文件。\n"
        "Suggestion: 请提供具体文件路径。"
    ),
    "syscall.file_read.binary_file": (
        "[Invalid Argument] '{path}' 看起来是二进制文件，无法作为文本读取。\n"
        "Suggestion: 请读取文本文件，或使用适合二进制内容的工具。"
    ),
    "syscall.file_read.read_failed": (
        "[Tool Error] 无法读取文件 '{path}'：{detail}\n"
        "Suggestion: 不要用相同输入重试该工具。"
    ),
    "syscall.file_write.missing_path": (
        "[Invalid Argument] file_write 缺少 \"{arg}\" 参数。\n"
        "Suggestion: 请提供要写入的工作区相对路径。"
    ),
    "syscall.file_write.missing_content": (
        "[Invalid Argument] file_write 缺少 \"{arg}\" 参数。\n"
        "Suggestion: 请提供要写入的内容。"
    ),
    "syscall.file_write.invalid_mode": (
        "[Invalid Argument] 写入模式 '{mode}' 无效，只支持 'overwrite' 或 'append'。\n"
        "Suggestion: 请修正 mode 参数后重试。"
    ),
    "syscall.file_write.content_too_large": (
        "[Invalid Argument] 内容过大（{size} bytes），最大允许 {max_bytes} bytes。\n"
        "Suggestion: 请缩短内容或拆分写入。"
    ),
    "syscall.file_write.path_denied": (
        "[Permission Denied] 路径 '{path}' 超出工作区边界。\n"
        "Suggestion: 请使用工作区内的相对路径。"
    ),
    "syscall.file_write.write_failed": (
        "[Tool Error] 无法写入文件 '{path}'：{detail}\n"
        "Suggestion: 不要用相同输入重试该工具。"
    ),
    "syscall.repl.missing_code": (
        "[Invalid Argument] python_repl 缺少 \"{arg}\" 参数。\n"
        "Suggestion: 请提供要执行的 Python 代码。"
    ),
    "syscall.repl.import_blocked": (
        "[Permission Denied] 受限 REPL 不允许 import。\n"
        "Suggestion: 只能使用沙箱允许的内置函数。"
    ),
    "syscall.repl.timeout": (
        "[Tool Error] Python 执行在 {timeout_seconds}s 后超时。\n"
        "Suggestion: 不要用相同输入重试该工具。"
    ),
    "syscall.repl.execution_failed": (
        "[Tool Error] Python 执行失败：{detail}\n"
        "Suggestion: 请检查代码中的运行时错误。"
    ),
    "syscall.web_search.missing_query": (
        "[Invalid Argument] web_search 缺少 \"{arg}\" 参数。\n"
        "Suggestion: 请提供搜索 query。"
    ),
    "syscall.web_search.unavailable": (
        "[Service Unavailable] 当前系统不可用 Web 搜索。\n"
        "Suggestion: 请改用其他方式完成任务。"
    ),
    "syscall.web_search.failed": (
        "[Tool Error] Web 搜索失败：{detail}\n"
        "Suggestion: 不要用相同输入重试该工具。"
    ),
}

_SYSCALL_ERROR_TEXT_EN: dict[str, str] = {
    "syscall.file_read.missing_path": (
        '[Invalid Argument] file_read requires a "{arg}" argument.\n'
        "Suggestion: Provide a workspace-relative path to read."
    ),
    "syscall.file_read.path_denied": (
        "[Permission Denied] Path '{path}' escapes the workspace boundary.\n"
        "Suggestion: Use a workspace-relative path."
    ),
    "syscall.file_read.not_found": (
        "[Invalid Argument] File not found: '{path}'.\n"
        "Suggestion: Check the path, or discover available files first."
    ),
    "syscall.file_read.not_file": (
        "[Invalid Argument] '{path}' is not a file.\n"
        "Suggestion: Provide a concrete file path."
    ),
    "syscall.file_read.binary_file": (
        "[Invalid Argument] '{path}' appears to be a binary file and cannot be read as text.\n"
        "Suggestion: Read a text file, or use a tool suited for binary content."
    ),
    "syscall.file_read.read_failed": (
        "[Tool Error] Cannot read file '{path}': {detail}\n"
        "Suggestion: Do NOT retry this tool with the same input."
    ),
    "syscall.file_write.missing_path": (
        '[Invalid Argument] file_write requires a "{arg}" argument.\n'
        "Suggestion: Provide a workspace-relative path to write."
    ),
    "syscall.file_write.missing_content": (
        '[Invalid Argument] file_write requires a "{arg}" argument.\n'
        "Suggestion: Provide the content to write."
    ),
    "syscall.file_write.invalid_mode": (
        "[Invalid Argument] Invalid write mode '{mode}'. Use 'overwrite' or 'append'.\n"
        "Suggestion: Fix the mode argument and retry."
    ),
    "syscall.file_write.content_too_large": (
        "[Invalid Argument] Content too large ({size} bytes). Maximum allowed: {max_bytes} bytes.\n"
        "Suggestion: Shorten the content or split the write."
    ),
    "syscall.file_write.path_denied": (
        "[Permission Denied] Path '{path}' escapes the workspace boundary.\n"
        "Suggestion: Use a workspace-relative path."
    ),
    "syscall.file_write.write_failed": (
        "[Tool Error] Cannot write file '{path}': {detail}\n"
        "Suggestion: Do NOT retry this tool with the same input."
    ),
    "syscall.repl.missing_code": (
        '[Invalid Argument] python_repl requires a "{arg}" argument.\n'
        "Suggestion: Provide Python code to execute."
    ),
    "syscall.repl.import_blocked": (
        "[Permission Denied] import is not allowed in the restricted REPL.\n"
        "Suggestion: Use only the built-in functions allowed by the sandbox."
    ),
    "syscall.repl.timeout": (
        "[Tool Error] Python execution timed out after {timeout_seconds}s.\n"
        "Suggestion: Do NOT retry this tool with the same input."
    ),
    "syscall.repl.execution_failed": (
        "[Tool Error] Python execution failed: {detail}\n"
        "Suggestion: Check your code for runtime errors."
    ),
    "syscall.web_search.missing_query": (
        '[Invalid Argument] web_search requires a "{arg}" argument.\n'
        "Suggestion: Provide a search query."
    ),
    "syscall.web_search.unavailable": (
        "[Service Unavailable] Web search is not available on this system.\n"
        "Suggestion: Use a different approach."
    ),
    "syscall.web_search.failed": (
        "[Tool Error] Web search failed: {detail}\n"
        "Suggestion: Do NOT retry this tool with the same input."
    ),
}

_SYSCALL_INFO_TEXT_ZH: dict[str, str] = {
    "syscall.repl.stdout": "Stdout: {output}",
    "syscall.repl.no_output": "执行成功（无输出）。",
    "syscall.file_read.truncated": "[Truncated: showing first {max_bytes} bytes of {file_size} bytes]",
    "syscall.file_write.success": "Success: File '{name}' saved ({bytes} bytes).",
    "syscall.web_search.no_results": "No results found for query: '{query}'",
    "syscall.web_search.result_item": "[{index}] Title: {title}\nSnippet: {snippet}\nURL: {url}",
}

_SYSCALL_INFO_TEXT_EN: dict[str, str] = {
    "syscall.repl.stdout": "Stdout: {output}",
    "syscall.repl.no_output": "Executed successfully (no output).",
    "syscall.file_read.truncated": "[Truncated: showing first {max_bytes} bytes of {file_size} bytes]",
    "syscall.file_write.success": "Success: File '{name}' saved ({bytes} bytes).",
    "syscall.web_search.no_results": "No results found for query: '{query}'",
    "syscall.web_search.result_item": "[{index}] Title: {title}\nSnippet: {snippet}\nURL: {url}",
}


def _get_syscall_runtime_text(
    key: str,
    params: dict[str, Any] | None,
    language: str | Language | None,
    *,
    zh_table: dict[str, str],
    en_table: dict[str, str],
    text_kind: str,
) -> str:
    lang = resolve_language(explicit=language)
    table = en_table if lang == Language.EN else zh_table
    try:
        template = table[key]
    except KeyError as exc:
        raise KeyError(f"Unknown syscall runtime {text_kind} i18n key: {key}") from exc

    try:
        return template.format(**(params or {}))
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Missing syscall runtime {text_kind} i18n param '{missing}' for key: {key}"
        ) from exc


def get_syscall_error_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """Return localized syscall error text."""
    return _get_syscall_runtime_text(
        key,
        params,
        language,
        zh_table=_SYSCALL_ERROR_TEXT_ZH,
        en_table=_SYSCALL_ERROR_TEXT_EN,
        text_kind="error",
    )


def get_syscall_info_text(
    key: str,
    params: dict[str, Any] | None = None,
    language: str | Language | None = None,
) -> str:
    """Return localized syscall success/info text."""
    return _get_syscall_runtime_text(
        key,
        params,
        language,
        zh_table=_SYSCALL_INFO_TEXT_ZH,
        en_table=_SYSCALL_INFO_TEXT_EN,
        text_kind="info",
    )


__all__ = ["get_syscall_error_text", "get_syscall_info_text"]
