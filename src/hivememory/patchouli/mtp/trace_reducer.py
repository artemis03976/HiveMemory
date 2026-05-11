"""
MTPTraceReducer — 兼容层：TurnEvent / AgentAction -> TraceItem

当前推荐链路已改为：
    TurnEvent -> ActionReducer -> AgentAction -> TraceReducer -> TraceItem

本类保留为兼容入口，内部委托新的摘要层实现。

作者: HiveMemory Team
版本: 1.0 (Phase 1)
"""

from typing import Any, Dict, List, Union

from hivememory.core.models import ActionReducer, AgentAction, TraceItem, TraceReducer


class MTPTraceReducer:
    """
    兼容转换器：接受 TurnEvent 列表或 AgentAction 列表，并委托 TraceReducer。
    """

    @classmethod
    def reduce(
        cls,
        items: List[Union[Any, Dict[str, Any]]],
    ) -> List[TraceItem]:
        """
        将 TurnEvent 或 AgentAction 列表转换为 TraceItem 列表。

        Args:
            items: TurnEvent / AgentAction 对象列表或等价的 dict 列表

        Returns:
            List[TraceItem]: 经过清洗的语义轨迹列表
        """
        if not items:
            return []

        first = items[0]
        if cls._looks_like_action(first):
            actions = [cls._normalize_action(item) for item in items]
        else:
            actions = ActionReducer.reduce(
                [cls._normalize_event(item, index) for index, item in enumerate(items)]
            )
        return TraceReducer.reduce(actions)

    @classmethod
    def _looks_like_action(cls, item: Union[Any, Dict[str, Any]]) -> bool:
        """粗略判断输入项是否已经是 AgentAction。"""
        if isinstance(item, dict):
            return "results" in item and "action_id" in item
        return hasattr(item, "results") and hasattr(item, "action_id")

    @classmethod
    def _normalize_action(cls, item: Union[Any, Dict[str, Any]]) -> AgentAction:
        """统一兼容对象与 dict 输入。"""
        if isinstance(item, AgentAction):
            return item
        if isinstance(item, dict):
            return AgentAction.model_validate(item)
        return AgentAction.model_validate(item.model_dump())

    @classmethod
    def _normalize_event(cls, item: Union[Any, Dict[str, Any]], index: int) -> Dict[str, Any]:
        """
        将旧 TurnEvent 输入归一化为适合 ActionReducer 的事件 dict。

        兼容行为：
        - 缺失 action_id 的 tool_call 使用 index 生成唯一键，避免旧测试/旧输入被错误合并
        - SEARCH 若缺失 tool_args，则尝试从 content 中提取 query
        """
        if isinstance(item, dict):
            event = dict(item)
        else:
            event = item.model_dump()

        kind = event.get("kind")
        if kind == "mtp_command":
            kind = "tool_call"
        elif kind == "mtp_result":
            kind = "tool_result"
        elif kind == "assistant_text":
            kind = "assistant_message"
        event["kind"] = kind

        tool_kind = (event.get("tool_kind") or event.get("verb") or "").upper()
        event["tool_kind"] = tool_kind or None

        if kind == "tool_call" and not event.get("action_id"):
            event["action_id"] = f"compat_action_{index}"

        if (
            kind == "tool_call"
            and tool_kind == "SEARCH"
            and not event.get("tool_args")
        ):
            query = cls._extract_search_query(event.get("content", "") or "")
            if query is not None:
                event["tool_args"] = {"query": query}

        return event

    @classmethod
    def _extract_search_query(cls, mtp_content: str) -> str | None:
        """从旧 MTP 文本中提取 SEARCH query，作为兼容回填。"""
        try:
            from hivememory.patchouli.mtp.parser import MTPParser

            parser = MTPParser()
            command = parser.parse(mtp_content)
            return command.args.get("query")
        except Exception:
            return None


__all__ = ["MTPTraceReducer"]
