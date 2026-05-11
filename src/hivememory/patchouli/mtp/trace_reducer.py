"""
MTPTraceReducer — TurnEvent 列表 → TraceItem 列表的结构化转换器

作为 MTPLogParser 文本解析的结构化替代路径。
当 InteractionPayload.turn_events 有值时，感知层使用此转换器
而非文本解析器，避免双重解析产生的噪音和歧义。

映射策略 (对齐 perception/models.py TraceItem 规范):
    mtp_command (READ)    → TraceItem(action="READ",   target=...)
    mtp_command (SEARCH)  → TraceItem(action="SEARCH", query=...)
    mtp_command (RUN)     → TraceItem(action="RUN",    tool=..., status=...)
    mtp_command (WRITE)   → 丢弃（作为控制信号处理，不生成 TraceItem）
    mtp_command (UPDATE)  → 丢弃（作为控制信号处理，不生成 TraceItem）
    mtp_command (CALL)    → 丢弃（子代理调用已在 koakuma._current_traces 记录）
    mtp_result            → 丢弃（结果消息不产生 TraceItem）
    assistant_text        → 丢弃（自然语言文本不产生 TraceItem）

RUN status 来源:
    TurnEvent.status 由 LoopExecutor 从 mtp_result.response_status 填充，
    直接传递给 TraceItem.status，无需重新解析文本。

作者: HiveMemory Team
版本: 1.0 (Phase 1)
"""

from typing import Any, Dict, List, Optional, Union

from hivememory.engines.perception.models import TraceItem


class MTPTraceReducer:
    """
    无状态转换器：将结构化 TurnEvent 列表化简为 TraceItem 列表。

    比 MTPLogParser 更精确，因为 verb/target/status 由 Koakuma 在执行时
    直接填充，而非事后从文本中正则提取。
    """

    @classmethod
    def reduce(
        cls,
        turn_events: List[Union[Any, Dict[str, Any]]],
    ) -> List[TraceItem]:
        """
        将 turn_events 转换为 TraceItem 列表。

        Args:
            turn_events: TurnEvent 对象列表或等价的 dict 列表
                         (chat_stream 路径经过 JSON 序列化后重建为 dict)

        Returns:
            List[TraceItem]: 经过清洗的语义轨迹列表
        """
        traces: List[TraceItem] = []
        for event in turn_events:
            trace = cls._event_to_trace(event)
            if trace is not None:
                traces.append(trace)
        return traces

    @classmethod
    def _event_to_trace(
        cls,
        event: Union[Any, Dict[str, Any]],
    ) -> Optional[TraceItem]:
        """将单个 event（对象或 dict）映射为 TraceItem，不适用则返回 None。"""
        # 统一读取：兼容 TurnEvent 对象和 dict（来自 SSE 序列化路径）
        if isinstance(event, dict):
            kind = event.get("kind", "")
            verb = (event.get("verb") or "").upper()
            target = event.get("target")
            status = event.get("status")
            content = event.get("content", "")
        else:
            kind = getattr(event, "kind", "")
            verb = (getattr(event, "verb", None) or "").upper()
            target = getattr(event, "target", None)
            status = getattr(event, "status", None)
            content = getattr(event, "content", "")

        # 只处理 mtp_command 类型
        if kind != "mtp_command":
            return None

        if verb == "READ":
            return TraceItem(action="READ", target=target)

        if verb == "SEARCH":
            query = cls._extract_search_query(content)
            return TraceItem(action="SEARCH", query=query)

        if verb == "RUN":
            return TraceItem(
                action="RUN",
                tool=target,
                status=status or "unknown",
            )

        # WRITE, UPDATE, CALL → 丢弃
        return None

    @classmethod
    def _extract_search_query(cls, mtp_content: str) -> Optional[str]:
        """
        从 MTP 原文中提取 SEARCH 的 query 参数。

        例: '⟪ SEARCH | * | query="docker config" ⟫'
        → 'docker config'

        委托给现有的 MTPParser 以复用解析逻辑，避免重复实现。
        解析失败时返回 None（TraceItem.query 为可选字段）。
        """
        try:
            from hivememory.patchouli.mtp.parser import MTPParser
            parser = MTPParser()
            command = parser.parse(mtp_content)
            return command.args.get("query")
        except Exception:
            return None


__all__ = ["MTPTraceReducer"]
