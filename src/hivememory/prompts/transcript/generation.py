"""
GenerationTranscriptBuilder — 记忆生成视图构建器

职责:
    从 LogicalBlock 列表构建"记忆生成视图"（GenerationContext）。
    该视图保留语义摘要与过程中的事实材料，供 GenerationEngine 提取记忆。

与 HistoryTranscriptBuilder 的区别:
    - HistoryTranscriptBuilder: 历史消息视图，还原模型当时看到的完整上下文
    - GenerationTranscriptBuilder: 记忆生成视图，按上限截断的过程材料，适合记忆提取

保留内容:
    - state_summary（话题状态摘要）
    - user_query
    - 动作段：思考、外部工具动作（工具名、参数摘要与截断后的结果）与
      MTP 动作的一行摘要，渲染规则见 ``prompts.transcript.actions``
    - assistant_final_text（最终自然语言回复）

丢弃内容:
    - MTP 动作的结果全文（READ 返回正文、SEARCH 结果体、XML 回填文本）
    - 超出上限的思考、参数与工具结果中段

动作段格式（action_entries）:
    - SEARCH: "query text"
    - READ: alias_name
    - RUN: tool_name (status)
    - THOUGHT: reasoning
    - TOOL: tool_name {"arg": "value"}
      RESULT (error): output
"""

from collections.abc import Sequence

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import LogicalBlock
from hivememory.engines.generation.models import GenerationContext, GenerationTurn
from hivememory.prompts.transcript.actions import render_action_entries


class GenerationTranscriptBuilder:
    """
    记忆生成视图构建器

    无状态。将 LogicalBlock 列表 + state_summary 转换为 GenerationContext，
    动作段按 ``config`` 的上限截断；同时提供 build_transcript() 将
    GenerationContext 渲染为 LLM 可用的文本（渲染不再截断）。
    """

    def __init__(self, config: GenerationTranscriptConfig | None = None) -> None:
        self._config = config or GenerationTranscriptConfig()

    def build_context(
        self,
        blocks: Sequence[LogicalBlock],
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
            - THOUGHT: ...
            - TOOL: Bash {"command": "pytest -q"}
              RESULT (error): ...
            - SEARCH: "..."
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
        return GenerationTurn(
            user_query=block.user_query,
            outcome=block.turn.outcome,
            assistant_final_text=block.assistant_final_text,
            action_entries=[
                entry.text for entry in render_action_entries(block.turn, self._config)
            ],
            identity=block.identity,
        )

    def _format_context(self, context: GenerationContext) -> str:
        """将 GenerationContext 渲染为文本。"""
        sections: list[str] = []

        if context.state_summary:
            sections.append(f"[Topic State]\n{context.state_summary}")

        for idx, turn in enumerate(context.turns, 1):
            sections.append(self._format_turn(idx, turn))

        return "\n\n".join(sections)

    def _format_turn(self, idx: int, turn: GenerationTurn) -> str:
        """渲染单轮对话；多行条目的后续行缩进到条目之下。"""
        lines = [f"[Turn {idx}]"]
        if turn.outcome != "completed":
            lines.append(f"[Outcome: {turn.outcome}] This turn did not complete normally.")

        if turn.user_query:
            lines.append(f"[User]: {turn.user_query}")

        if turn.action_entries:
            lines.append("[Actions]:")
            for entry in turn.action_entries:
                first, *rest = entry.split("\n")
                lines.append(f"- {first}")
                lines.extend(f"  {line}" for line in rest)

        if turn.assistant_final_text:
            lines.append(f"[Assistant]: {turn.assistant_final_text}")

        return "\n".join(lines)


__all__ = ["GenerationTranscriptBuilder"]
