"""
GenerationTranscriptBuilder — 记忆生成视图构建器

职责:
    从 LogicalBlock 列表构建"记忆生成视图"（GenerationContext）。
    该视图保留语义摘要与过程中的事实材料，供 GenerationEngine 提取记忆。

与 HistoryTranscriptBuilder 的区别:
    - HistoryTranscriptBuilder: 历史消息视图，还原模型当时看到的完整上下文
    - GenerationTranscriptBuilder: 记忆生成视图，按上限截断的过程材料，适合记忆提取

每轮的材料由感知层唯一的渲染函数从事件集得到（``engines.perception.material``）：
助手正文与动作按出现顺序穿插，能力调用只有一行摘要、不渲染取回的记忆，思考、
外部工具参数与结果按上限截断。这里只负责组装话题状态与各轮次。
"""

from collections.abc import Sequence

from hivememory.config.patchouli import GenerationTranscriptConfig
from hivememory.core.models import LogicalBlock
from hivememory.engines.generation.models import GenerationContext, GenerationTurn
from hivememory.engines.perception.material import format_turn_material, render_turn_material


class GenerationTranscriptBuilder:
    """
    记忆生成视图构建器

    无状态。将 LogicalBlock 列表 + state_summary 转换为 GenerationContext，
    材料按 ``config`` 的上限截断；同时提供 build_transcript() 将
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
        return GenerationContext(
            state_summary=state_summary,
            turns=[self._block_to_turn(block) for block in blocks],
        )

    def build_transcript(self, context: GenerationContext) -> str:
        """
        将 GenerationContext 渲染为 LLM 可用的文本（送入 {transcript} 占位符）。

        格式:
            [Topic State]
            {state_summary}

            [Turn 1]
            [User]: ...
            - THOUGHT: ...
            - SEARCH: "..."
            [Assistant]: ...
            - TOOL: Bash {"command": "pytest -q"}
              RESULT (error): ...
            [Assistant]: ...

        Args:
            context: GenerationContext 对象

        Returns:
            str: 渲染后的 transcript 文本
        """
        sections: list[str] = []
        if context.state_summary:
            sections.append(f"[Topic State]\n{context.state_summary}")
        for idx, turn in enumerate(context.turns, 1):
            sections.append(self._format_turn(idx, turn))
        return "\n\n".join(sections)

    # ============ 内部方法 ============

    def _block_to_turn(self, block: LogicalBlock) -> GenerationTurn:
        """将单个 LogicalBlock 转换为 GenerationTurn。"""
        return GenerationTurn(
            outcome=block.turn.outcome,
            user_message=block.user_message,
            material_entries=[
                entry.text for entry in render_turn_material(block.turn_events, self._config)
            ],
            identity=block.identity,
        )

    @staticmethod
    def _format_turn(idx: int, turn: GenerationTurn) -> str:
        """渲染单轮材料；结局不是 completed 时在轮首标注。"""
        lines = [f"[Turn {idx}]"]
        if turn.outcome != "completed":
            lines.append(f"[Outcome: {turn.outcome}] This turn did not complete normally.")
        lines.append(format_turn_material(turn.user_message, turn.material_entries))
        return "\n".join(lines)


__all__ = ["GenerationTranscriptBuilder"]
