"""
网络搜索 syscall。
"""

from typing import Dict


def sys_web_search(args: Dict[str, str], *, timeout_seconds: int = 15) -> str:
    """
    网络搜索 (Chapter 8.2)。

    timeout_seconds 参数为兼容注册表保留，当前实现未直接使用。
    """
    _ = timeout_seconds
    query = args.get("query", "")
    if not query:
        return "Error: 'query' argument is required."

    num_str = args.get("num", "3")
    try:
        num = max(1, min(10, int(num_str)))
    except (ValueError, TypeError):
        num = 3

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: Web search is not available on this system. Use a different approach."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num))
    except Exception:
        return "Error: Web search failed. The search service may be temporarily unavailable."

    if not results:
        return f"No results found for query: '{query}'"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "N/A")
        snippet = r.get("body", "N/A")
        url = r.get("href", "N/A")
        lines.append(f"[{i}] Title: {title}\nSnippet: {snippet}\nURL: {url}")

    return "\n\n".join(lines)
