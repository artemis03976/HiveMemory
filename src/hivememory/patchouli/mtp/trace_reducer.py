"""
MTPTraceReducer — 轻量门面：TurnEvent / AgentAction -> TraceItem

当前推荐链路已改为：
    TurnEvent -> ActionReducer -> AgentAction -> TraceReducer -> TraceItem

本类仅保留输入类型适配，内部委托新的摘要层实现。

作者: HiveMemory Team
版本: 1.0 (Phase 1)
"""

from typing import Any, Dict, List, Union

from hivememory.core.models import ActionReducer, AgentAction, TraceItem, TraceReducer


class MTPTraceReducer:
    """
    轻量转换器：接受 TurnEvent 列表或 AgentAction 列表，并委托 TraceReducer。
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
            actions = ActionReducer.reduce(items)
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


__all__ = ["MTPTraceReducer"]
