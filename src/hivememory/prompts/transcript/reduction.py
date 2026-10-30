"""历史工具结果的无模型减重，只生成派生文本，不修改轮次事件。"""

from hivememory.core.models import TurnEvent
from hivememory.i18n import get_compaction_prompt_text


def reduce_tool_result(
    call: TurnEvent,
    *,
    language: str | None,
    excerpt_chars: int,
) -> str:
    """成功的可重取结果改为线索，其余结果保留有界原文；改写结果不会比原文更长。

    ``call`` 是有结局的调用事件，动词、作用对象、参数与返回原文都在它上面。
    """
    result = call.result
    if result is None:
        raise ValueError(f"tool_call has no outcome to reduce: {call.action_id}")
    rewritten = _retrieval_hint(call, result, language=language, excerpt_chars=excerpt_chars)
    # 原文本身比线索还短时，改写只会让历史变长。
    if rewritten is not None and len(rewritten) < len(result):
        return rewritten
    return _bounded_excerpt(result, excerpt_chars, language)


def _retrieval_hint(
    call: TurnEvent, result: str, *, language: str | None, excerpt_chars: int
) -> str | None:
    """只为成功的 READ/SEARCH/RUN 生成线索；错误与未知状态的正文含恢复所需原因。

    CALL 回填含子 Agent 的回复，不可重取，按动词识别后保留有界原文。
    """
    if call.status != "success":
        return None
    target = ",".join(call.targets)
    if call.verb == "READ" and target:
        hint = get_compaction_prompt_text("hint_read_again", language)
        return f"[READ target={target} status=success]\n{hint}"
    if call.verb == "SEARCH":
        query = call.tool_args.get("query") if call.tool_args is not None else None
        if query:
            hint = get_compaction_prompt_text("hint_search_again", language)
            return f"[SEARCH query={query} status=success]\n{hint}"
    if call.verb == "RUN":
        # 参数随调用消息原样保留在历史中，结果里不再重复，避免大参数让减重后的历史变长。
        return (
            f"[RUN tool={target} status=success]\n"
            f"{_bounded_excerpt(result, excerpt_chars, language)}"
        )
    return None


def _bounded_excerpt(content: str, limit: int, language: str | None) -> str:
    """保留头尾并标记省略，避免长错误或 ACK 的尾部信息全部丢失。"""
    if len(content) <= limit:
        return content
    marker = get_compaction_prompt_text("excerpt_omitted", language)
    if limit <= len(marker):
        return marker[:limit]
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available // 2
    return content[:head] + marker + (content[-tail:] if tail else "")
