"""
GenerationTranscriptBuilder — 记忆生成视图构建器

职责:
    从 LogicalBlock 列表构建"记忆生成视图"（GenerationContext）。
    该视图保留语义摘要，丢弃 MTP 执行细节，供 GenerationEngine 提取记忆。

与 HistoryTranscriptBuilder 的区别:
    - HistoryTranscriptBuilder: 历史消息视图，还原模型当时看到的完整上下文
    - GenerationTranscriptBuilder: 记忆生成视图，去噪后的语义摘要，适合记忆提取

保留内容:
    - state_summary（话题状态摘要）
    - user_query
    - semantic_traces（动作摘要）
    - assistant_final_text（最终自然语言回复）

丢弃内容:
    - mtp_result 全文（READ 返回正文、SEARCH 结果体、XML 回填文本）
    - 完整的 turn_events 事件流

动作摘要格式（trace_summaries）:
    - SEARCH: "query text"
    - READ: alias_name
    - RUN: tool_name (status)

作者: HiveMemory Team
版本: 1.0 (Phase 3)
"""

from typing import List, Optional

from hivememory.core.constants import DEFAULT_AGENT_ID, DEFAULT_USER_ID
from hivememory.core.models import Identity, TraceItem
from hivememory.engines.perception.models import LogicalBlock
from hivememory.engines.generation.models import GenerationContext, GenerationTurn


class GenerationTranscriptBuilder:
    """
    记忆生成视图构建器

    无状态。将 LogicalBlock 列表 + state_summary 转换为 GenerationContext。
    同时提供 build_transcript() 将 GenerationContext 渲染为 LLM 可用的文本。
    """

    def build_context(
        self,
        blocks: List[LogicalBlock],
        state_summary: str = "",
    ) -> GenerationContext:
        """
        从 LogicalBlock 列表构建 GenerationContext。

        Args:
            blocks: LogicalBlock 列表（按时间正序）
            state_summary: 话题状态摘要

        Returns:
            GenerationContext: 结构化的记忆生成视图
        """
        turns = [self._block_to_turn(block) for block in blocks]
        # 过滤掉既无 final_text 又无 user_query 的空轮次
        turns = [t for t in turns if t.user_query or t.assistant_final_text]
        return GenerationContext(state_summary=state_summary, turns=turns)

    def build_transcript(self, context: GenerationContext) -> str:
        """
        将 GenerationContext 渲染为 LLM 可用的文本（送入 {transcript} 占位符）。

        格式:
            [Topic State]
            {state_summary}

            [Turn 1]
            [User]: ...
            [Actions]:
            - SEARCH: "..."
            - READ: alias_x
            [Assistant]: ...

        Args:
            context: GenerationContext 对象

        Returns:
            str: 渲染后的 transcript 文本
        """
        return self._format_context(context)

    # ============ 内部方法 ============

    def _block_to_turn(self, block: LogicalBlock) -> GenerationTurn:
        """将单个 LogicalBlock 转换为 GenerationTurn。"""
        user_query = block.user_query
        final_text = block.assistant_final_text
        identity = block.identity

        # 动作摘要：优先从 semantic_traces 派生
        trace_summaries = self._traces_to_summaries(block.semantic_traces)

        return GenerationTurn(
            user_query=user_query,
            assistant_final_text=final_text,
            trace_summaries=trace_summaries,
            identity=identity,
        )

    def _traces_to_summaries(self, traces: List[TraceItem]) -> List[str]:
        """将 TraceItem 列表转换为可读的动作摘要字符串列表。"""
        summaries = []
        for trace in traces:
            summary = self._trace_to_summary(trace)
            if summary:
                summaries.append(summary)
        return summaries

    def _trace_to_summary(self, trace: TraceItem) -> Optional[str]:
        """将单个 TraceItem 转换为动作摘要字符串。"""
        action = trace.action.upper()

        if action == "SEARCH":
            query = trace.query or ""
            return f'SEARCH: "{query}"'

        if action == "READ":
            target = trace.target or "(unknown)"
            return f"READ: {target}"

        if action == "RUN":
            tool = trace.tool or "(unknown)"
            status = trace.status or "unknown"
            return f"RUN: {tool} ({status})"

        # CALL / WRITE / UPDATE remain in semantic_traces but are not rendered into generation summaries.
        return None

    def _format_context(self, context: GenerationContext) -> str:
        """将 GenerationContext 渲染为文本。"""
        sections: List[str] = []

        if context.state_summary:
            sections.append(f"[Topic State]\n{context.state_summary}")

        for idx, turn in enumerate(context.turns, 1):
            sections.append(self._format_turn(idx, turn))

        return "\n\n".join(sections)

    def _format_turn(self, idx: int, turn: GenerationTurn) -> str:
        """渲染单轮对话。"""
        lines = [f"[Turn {idx}]"]

        if turn.user_query:
            lines.append(f"[User]: {turn.user_query}")

        if turn.trace_summaries:
            lines.append("[Actions]:")
            for summary in turn.trace_summaries:
                lines.append(f"- {summary}")

        if turn.assistant_final_text:
            lines.append(f"[Assistant]: {turn.assistant_final_text}")

        return "\n".join(lines)


__all__ = ["GenerationTranscriptBuilder"]
