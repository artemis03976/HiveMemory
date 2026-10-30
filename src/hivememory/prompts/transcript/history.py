"""会话历史转写：按全量轮次重放每一轮的事件。"""

from collections.abc import Sequence

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import ConversationTurn
from hivememory.i18n import get_compaction_prompt_text
from hivememory.prompts.transcript.reduction import reduce_tool_result


class HistoryTranscriptBuilder:
    """会话轮次生成下一轮 Agent 的历史消息，工具结果沿用既有渲染规则。"""

    def build_messages(
        self,
        turns: Sequence[ConversationTurn],
        current_agent_id: str = "default",
        language: str | None = None,
        *,
        reduced_through: str | None = None,
        tool_result_excerpt_chars: int = 2000,
        tool_result_max_chars: int | None = None,
    ) -> list[dict[str, str]]:
        """
        构建历史消息列表。

        Args:
            turns: 会话轮次列表（按时间正序）
            current_agent_id: 当前活跃 Agent 别名，用于多角色前缀判断
            language: 结果线索与未完成轮次说明使用的提示词语言
            reduced_through: 包含该轮次在内的工具结果减重边界
            tool_result_excerpt_chars: 保留型工具结果的正文字符上限
            tool_result_max_chars: 摘要输入可另限制完整工具消息，含参数与包装前缀

        Returns:
            OpenAI-style messages: List[{"role": str, "content": str}]
        """
        messages: list[dict[str, str]] = []

        boundary = next(
            (index for index, turn in enumerate(turns) if turn.interaction_id == reduced_through),
            -1,
        )
        for index, turn in enumerate(turns):
            start = len(messages)
            self._render_turn(
                turn,
                current_agent_id,
                messages,
                reduce_results=index <= boundary,
                language=language,
                excerpt_chars=tool_result_excerpt_chars,
                result_max_chars=tool_result_max_chars,
            )
            note = self._outcome_note(turn.payload.outcome, language)
            if note and len(messages) > start:
                # 结局说明附在本轮最后一条消息末尾，不在历史中间插入 system 消息：
                # 部分模型适配会把 system 消息合并到开头，或要求角色严格交替。
                last = messages[-1]
                messages[-1] = {**last, "content": f"{last['content']}\n\n{note}"}

        return messages

    # ============ 内部方法 ============

    @staticmethod
    def _outcome_note(outcome: str, language: str | None) -> str:
        """未正常完成的轮次给出结局说明；completed 不加说明。"""
        if outcome == "completed":
            return ""
        key = "outcome_cancelled" if outcome == "cancelled" else "outcome_failed"
        return get_compaction_prompt_text(key, language)

    def _render_turn(
        self,
        turn: ConversationTurn,
        current_agent_id: str,
        out: list[dict[str, str]],
        *,
        reduce_results: bool,
        language: str | None,
        excerpt_chars: int,
        result_max_chars: int | None,
    ) -> None:
        """按事件顺序重放单个会话轮次，追加到 out。

        角色由事件决定：用户消息与工具调用的结局进入 user 消息（工具返回放进
        user 消息是 Alice 的重放规则，与运行时回填一致），其余进入 assistant
        消息。指令前正文与紧随其后的 MTP 调用合并成一条助手消息，与运行时
        working history 中的形态一致，指令前正文因此只出现一次。回填正文自带
        标题，重放原样使用；没有结局的调用只重放调用原文。
        """
        previous: TurnEvent | None = None
        for event in turn.payload.turn_events:
            if event.kind == "user_message":
                out.append({"role": "user", "content": event.content})
            elif (
                event.kind == "tool_call"
                and previous is not None
                and previous.kind == "assistant_message"
            ):
                out[-1] = {**out[-1], "content": out[-1]["content"] + event.content}
            else:
                content = self._apply_agent_prefix(event.content, turn, current_agent_id)
                out.append({"role": "assistant", "content": content})
            if event.kind == "tool_call" and event.result is not None:
                result = event.result
                if reduce_results:
                    result = reduce_tool_result(
                        event, language=language, excerpt_chars=excerpt_chars
                    )
                if result_max_chars is not None:
                    result = result[:result_max_chars]
                out.append({"role": "user", "content": result})
            previous = event

    def _apply_agent_prefix(
        self,
        content: str,
        turn: ConversationTurn,
        current_agent_id: str,
    ) -> str:
        """若轮次来自非当前 Agent，在内容前加 [From: {agent_id}]。"""
        agent_id = turn.agent_id
        if agent_id and agent_id != current_agent_id:
            return f"[From: {agent_id}]\n{content}"
        return content


__all__ = ["HistoryTranscriptBuilder"]
