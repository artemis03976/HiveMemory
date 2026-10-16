"""会话历史转写：按全量轮次重放事件，缺少事件时回退到回复文本。"""

from collections.abc import Sequence

from hivememory.core.models import TurnEvent
from hivememory.core.protocol.models import ConversationTurn
from hivememory.i18n.mtp_runtime import get_mtp_info_text

_SYSTEM_PREFIX_KEYS: dict[str, str] = {
    "system_tool_result": "mtp.loop.execution_result_title",
    "system_call_response": "mtp.call_response.title",
}


class HistoryTranscriptBuilder:
    """会话轮次生成下一轮 Agent 的历史消息，工具结果沿用既有渲染规则。"""

    def build_messages(
        self,
        turns: Sequence[ConversationTurn],
        current_agent_id: str = "default",
    ) -> list[dict[str, str]]:
        """
        构建历史消息列表。

        Args:
            turns: 会话轮次列表（按时间正序）
            current_agent_id: 当前活跃 Agent 别名，用于多角色前缀判断

        Returns:
            OpenAI-style messages: List[{"role": str, "content": str}]
        """
        messages: list[dict[str, str]] = []

        for turn in turns:
            self._render_turn(turn, current_agent_id, messages)

        return messages

    # ============ 内部方法 ============

    def _render_turn(
        self,
        turn: ConversationTurn,
        current_agent_id: str,
        out: list[dict[str, str]],
    ) -> None:
        """渲染单个会话轮次，追加到 out。"""
        payload = turn.payload
        if payload.turn_events:
            # 结构化路径：按 sequence 重放事件流
            if payload.user_message and not self._has_user_message_event(payload.turn_events):
                out.append({"role": "user", "content": payload.user_message})
            for event in sorted(payload.turn_events, key=lambda e: e.sequence):
                msg = self._render_event(event, turn, current_agent_id)
                if msg is not None:
                    out.append(msg)
        else:
            user_content = payload.user_message
            if user_content:
                out.append({"role": "user", "content": user_content})
            assistant_content = payload.assistant_final_text
            if not assistant_content:
                return
            content = self._apply_agent_prefix(assistant_content, turn, current_agent_id)
            out.append({"role": "assistant", "content": content})

    def _render_event(
        self,
        event: TurnEvent,
        turn: ConversationTurn,
        current_agent_id: str,
    ) -> "dict[str, str] | None":
        """
        将单个 TurnEvent 转换为 OpenAI message dict。

        Returns:
            dict 或 None（未来可用于过滤特定事件类型）
        """
        content = self._apply_system_prefix(event)

        if event.role == "assistant":
            content = self._apply_agent_prefix(content, turn, current_agent_id)

        return {"role": event.role, "content": content}

    def _has_user_message_event(self, events: Sequence[TurnEvent]) -> bool:
        """检查结构化事件流是否已经承载用户消息，避免重放时重复补 user_query。"""
        return any(event.kind == "user_message" for event in events)

    def _apply_system_prefix(self, event: TurnEvent) -> str:
        """根据 render_as 为 mtp_result 类事件补系统前缀。"""
        render_as = event.render_as
        prefix_key = _SYSTEM_PREFIX_KEYS.get(render_as)
        prefix = get_mtp_info_text(prefix_key) if prefix_key else None
        if prefix:
            if event.content == prefix or event.content.startswith(f"{prefix}\n"):
                return event.content
            return f"{prefix}\n{event.content}"
        return event.content

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
