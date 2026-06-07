"""
网络搜索 syscall。
"""

from typing import Dict

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult
from hivememory.core.mtp.exceptions import (
    SyscallExecutionError,
    SyscallInvalidArgumentError,
    SyscallUnavailableError,
)
from hivememory.i18n.syscall_runtime import get_syscall_info_text


def sys_web_search(args: Dict[str, str], *, timeout_seconds: int = 15) -> SyscallResult:
    """
    网络搜索 (Chapter 8.2)。

    timeout_seconds 参数为兼容注册表保留，当前实现未直接使用。
    """
    _ = timeout_seconds
    query = args.get("query", "")
    if not query:
        raise SyscallInvalidArgumentError(
            message_key="syscall.web_search.missing_query",
            params={"arg": "query"},
        )

    num_str = args.get("num", "3")
    try:
        num = max(1, min(10, int(num_str)))
    except (ValueError, TypeError):
        num = 3

    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise SyscallUnavailableError(
            message_key="syscall.web_search.unavailable",
            cause=exc,
        ) from exc

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num))
    except Exception as exc:
        raise SyscallExecutionError(
            message_key="syscall.web_search.failed",
            params={"detail": "The search service may be temporarily unavailable."},
            cause=exc,
        ) from exc

    if not results:
        return SyscallResult(
            content=get_syscall_info_text(
                "syscall.web_search.no_results",
                {"query": query},
            )
        )

    lines = []
    field_empty = get_syscall_info_text("syscall.web_search.field_empty")
    for i, r in enumerate(results, 1):
        title = r.get("title") or field_empty
        snippet = r.get("body") or field_empty
        url = r.get("href") or field_empty
        lines.append(
            get_syscall_info_text(
                "syscall.web_search.result_item",
                {"index": i, "title": title, "snippet": snippet, "url": url},
            )
        )

    return SyscallResult(content="\n\n".join(lines))
