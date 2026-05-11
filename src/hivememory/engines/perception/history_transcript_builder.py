"""
HistoryTranscriptBuilder — 基于 TurnEvent 事件流的历史消息视图构建器

职责:
    从 LogicalBlock 列表构建"历史消息视图"（用于下轮 Agent 对话上下文）。
    - 若 block 有 turn_events，按 sequence 顺序重放结构化事件
    - 若 block 无 turn_events，优先回退到 assistant_final_text
    - 若为真正的 legacy block，再薄兼容到 user_block/response_block

渲染规则:
    - assistant_text  → role="assistant", content 原样输出（可附身份前缀）
    - mtp_command     → role="assistant", content 原样输出（可附身份前缀）
    - mtp_result      → role="user", 按 render_as 补系统前缀（不附身份前缀）

render_as 前缀策略:
    - "plain"               → 原样输出 content
    - "system_mtp_result"   → "[System MTP Execution Result]\\n{content}"
    - "system_ipc_return"   → "[System IPC Return]\\n{content}"

多智能体身份前缀:
    对 role="assistant" 的事件，若 block.identity.agent_id 不是
    "default" / "omni_doll" 且与 current_agent_id 不同，则在内容前追加：
        "[From: {agent_id}]\\n"

作者: HiveMemory Team
版本: 1.0 (Phase 2)
"""

from typing import Dict, List

from hivememory.engines.perception.models import LogicalBlock, TurnEvent


_SYSTEM_PREFIXES: Dict[str, str] = {
    "system_mtp_result": "[System MTP Execution Result]",
    "system_ipc_return": "[System IPC Return]",
}

_AGENT_ID_BYPASS = frozenset({"default", "omni_doll"})


class HistoryTranscriptBuilder:
    """
    历史消息视图构建器

    从 LogicalBlock 列表生成可直接送入下轮 LLM 的 OpenAI-style messages。
    优先消费 block.turn_events，其次使用 assistant_final_text，
    最后薄兼容 legacy user_block/response_block。
    """

    def build_messages(
        self,
        blocks: List[LogicalBlock],
        current_agent_id: str = "default",
    ) -> List[Dict[str, str]]:
        """
        构建历史消息列表。

        Args:
            blocks: LogicalBlock 列表（按时间正序）
            current_agent_id: 当前活跃 Agent 别名，用于多角色前缀判断

        Returns:
            OpenAI-style messages: List[{"role": str, "content": str}]
        """
        messages: List[Dict[str, str]] = []

        for block in blocks:
            self._render_block(block, current_agent_id, messages)

        return messages

    # ============ 内部方法 ============

    def _render_block(
        self,
        block: LogicalBlock,
        current_agent_id: str,
        out: List[Dict[str, str]],
    ) -> None:
        """渲染单个 LogicalBlock，追加到 out。"""
        user_content = self._resolve_user_content(block)
        if user_content:
            out.append({"role": "user", "content": user_content})

        if block.turn_events:
            # 结构化路径：按 sequence 重放事件流
            for event in sorted(block.turn_events, key=lambda e: e.sequence):
                msg = self._render_event(event, block, current_agent_id)
                if msg is not None:
                    out.append(msg)
        else:
            assistant_content = self._resolve_assistant_content(block)
            if not assistant_content:
                return
            content = self._apply_agent_prefix(
                assistant_content, block, current_agent_id
            )
            out.append({"role": "assistant", "content": content})

    def _resolve_user_content(self, block: LogicalBlock) -> str:
        """解析历史视图中的用户消息内容。"""
        if block.user_query:
            return block.user_query
        if block.user_block:
            return block.user_block.content
        return ""

    def _resolve_assistant_content(self, block: LogicalBlock) -> str:
        """解析历史视图中的 assistant 内容，逐步移除对 clean_response 的依赖。"""
        if block.assistant_final_text:
            return block.assistant_final_text
        if block.response_block:
            return block.response_block.content
        return ""

    def _render_event(
        self,
        event: TurnEvent,
        block: LogicalBlock,
        current_agent_id: str,
    ) -> "Dict[str, str] | None":
        """
        将单个 TurnEvent 转换为 OpenAI message dict。

        Returns:
            dict 或 None（未来可用于过滤特定事件类型）
        """
        content = self._apply_system_prefix(event)

        if event.role == "assistant":
            content = self._apply_agent_prefix(content, block, current_agent_id)

        return {"role": event.role, "content": content}

    def _apply_system_prefix(self, event: TurnEvent) -> str:
        """根据 render_as 为 mtp_result 类事件补系统前缀。"""
        render_as = event.render_as
        prefix = _SYSTEM_PREFIXES.get(render_as)
        if prefix:
            return f"{prefix}\n{event.content}"
        return event.content

    def _apply_agent_prefix(
        self,
        content: str,
        block: LogicalBlock,
        current_agent_id: str,
    ) -> str:
        """若 block 来自非当前 Agent，在内容前加 [From: {agent_id}]。"""
        agent_id = block.identity.agent_id
        if (
            agent_id
            and agent_id not in _AGENT_ID_BYPASS
            and agent_id != current_agent_id
        ):
            return f"[From: {agent_id}]\n{content}"
        return content


__all__ = ["HistoryTranscriptBuilder"]
