"""历史工具结果的无模型减重，只生成派生文本，不修改轮次事件。"""

import json

from hivememory.core.models import TurnEvent
from hivememory.i18n import resolve_language


def reduce_tool_result(
    event: TurnEvent,
    call: TurnEvent | None,
    *,
    language: str | None,
    excerpt_chars: int,
) -> str:
    """成功的可重取结果保留线索，其余结果保留有界原文。"""
    zh = resolve_language(explicit=language).value == "zh"
    verb = (event.tool_kind or (call.tool_kind if call else "") or "").upper()
    target = (
        event.target or event.tool_name or ((call.target or call.tool_name) if call else "") or ""
    )
    args = event.tool_args if event.tool_args is not None else (call.tool_args if call else None)
    status = event.status or "unknown"
    # 错误正文包含恢复所需的原因；未知状态也不假定为成功。
    if status == "success" and event.render_as != "system_call_response":
        if verb == "READ" and target:
            hint = "可重新 READ 获取正文。" if zh else "READ again to retrieve the content."
            return f"[READ target={target} status={status}]\n{hint}"
        if verb == "SEARCH":
            query = args.get("query", target) if args is not None else target
            if query:
                hint = "可使用原查询重新检索。" if zh else "Search again with the original query."
                return f"[SEARCH query={query} status={status}]\n{hint}"
        if verb == "RUN":
            arguments = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
            return (
                f"[RUN tool={target} args={arguments} status={status}]\n"
                f"{_bounded_excerpt(event.content, excerpt_chars, zh)}"
            )
    return _bounded_excerpt(event.content, excerpt_chars, zh)


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
