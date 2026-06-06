"""
网络搜索 syscall。
"""

from typing import Dict

from hivememory.agent_runtime.mtp.syscalls.types import SyscallResult


def sys_web_search(args: Dict[str, str], *, timeout_seconds: int = 15) -> SyscallResult:
    """
    网络搜索 (Chapter 8.2)。

    timeout_seconds 参数为兼容注册表保留，当前实现未直接使用。
    """
    _ = timeout_seconds
    query = args.get("query", "")
    if not query:
        return SyscallResult(ok=False, content="'query' argument is required.", error_code="mtp.argument.invalid")

    num_str = args.get("num", "3")
    try:
        num = max(1, min(10, int(num_str)))
    except (ValueError, TypeError):
        num = 3

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return SyscallResult(ok=False, content="Web search is not available on this system. Use a different approach.", error_code="mtp.system.tool_error")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num))
    except Exception:
        return SyscallResult(ok=False, content="Web search failed. The search service may be temporarily unavailable.", error_code="mtp.system.tool_error")

    if not results:
        return SyscallResult(ok=True, content=f"No results found for query: '{query}'")

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "N/A")
        snippet = r.get("body", "N/A")
        url = r.get("href", "N/A")
        lines.append(f"[{i}] Title: {title}\nSnippet: {snippet}\nURL: {url}")

    return SyscallResult(ok=True, content="\n\n".join(lines))
