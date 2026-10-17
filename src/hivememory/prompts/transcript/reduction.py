"""历史工具结果的无模型减重，只生成派生文本，不修改轮次事件。"""

from hivememory.core.models import TurnEvent
from hivememory.i18n import resolve_language


def reduce_tool_result(
    event: TurnEvent,
    call: TurnEvent | None,
    *,
    language: str | None,
    excerpt_chars: int,
) -> str:
    """成功的可重取结果改为线索，其余结果保留有界原文；改写结果不会比原文更长。"""
    zh = resolve_language(explicit=language).value == "zh"
    rewritten = _retrieval_hint(event, call, zh=zh, excerpt_chars=excerpt_chars)
    # 原文本身比线索还短时，改写只会让历史变长。
    if rewritten is not None and len(rewritten) < len(event.content):
        return rewritten
    return _bounded_excerpt(event.content, excerpt_chars, zh)


def _retrieval_hint(
    event: TurnEvent, call: TurnEvent | None, *, zh: bool, excerpt_chars: int
) -> str | None:
    """只为成功的 READ/SEARCH/RUN 生成线索；错误与未知状态的正文含恢复所需原因。"""
    if event.status != "success" or event.render_as == "system_call_response":
        return None
    verb = (event.tool_kind or (call.tool_kind if call else "") or "").upper()
    target = (
        event.target or event.tool_name or ((call.target or call.tool_name) if call else "") or ""
    )
    if verb == "READ" and target:
        hint = "可重新 READ 获取正文。" if zh else "READ again to retrieve the content."
        return f"[READ target={target} status=success]\n{hint}"
    if verb == "SEARCH":
        args = (
            event.tool_args if event.tool_args is not None else (call.tool_args if call else None)
        )
        query = args.get("query", target) if args is not None else target
        if query:
            hint = "可使用原查询重新检索。" if zh else "Search again with the original query."
            return f"[SEARCH query={query} status=success]\n{hint}"
    if verb == "RUN":
        # 参数随调用消息原样保留在历史中，结果里不再重复，避免大参数让减重后的历史变长。
        return (
            f"[RUN tool={target} status=success]\n"
            f"{_bounded_excerpt(event.content, excerpt_chars, zh)}"
        )
    return None


def _bounded_excerpt(content: str, limit: int, zh: bool) -> str:
    """保留头尾并标记省略，避免长错误或 ACK 的尾部信息全部丢失。"""
    if len(content) <= limit:
        return content
    marker = "\n[已省略部分工具结果]\n" if zh else "\n[Tool result excerpt omitted]\n"
    if limit <= len(marker):
        return marker[:limit]
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available // 2
    return content[:head] + marker + (content[-tail:] if tail else "")
