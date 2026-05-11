"""
HiveMemory 感知层上下文转换器

将感知层数据转换为 LLM 可用的格式。

职责:
    - 将 LogicalBlock 列表转换为 OpenAI messages 格式
    - 将 TopicSnapshot 列表转换为文本格式供 TheEye 使用

作者: HiveMemory Team
版本: 2.0.0 (Phase 2: 历史视图委托 HistoryTranscriptBuilder)
"""

from typing import List, Dict
from hivememory.engines.perception.models import LogicalBlock, TopicSnapshot
from hivememory.engines.perception.history_transcript_builder import HistoryTranscriptBuilder

_builder = HistoryTranscriptBuilder()


class PerceptionContextConverter:
    """将感知层数据转换为 LLM 可用的格式"""

    @staticmethod
    def blocks_to_messages(
        blocks: List[LogicalBlock],
        include_state_summary: bool = False,
        state_summary: str = "",
        current_agent_id: str = "default",
    ) -> List[Dict[str, str]]:
        """
        将 LogicalBlock 列表转换为 OpenAI messages 格式

        Phase 2 重构：委托 HistoryTranscriptBuilder 处理历史消息视图。
        - 若 block 有 turn_events，按事件流重放（保留 MTP 指令与结果）
        - 若 block 无 turn_events，回退到 clean_response（兼容旧数据）

        多角色历史渲染 (Phase 1):
        当 block.identity.agent_id 与 current_agent_id 不同时，
        在 assistant 消息头部追加身份标识前缀，防止"认领幻觉"。

        Args:
            blocks: LogicalBlock 列表
            include_state_summary: 已废弃，保留用于兼容
            state_summary: 已废弃，保留用于兼容
            current_agent_id: 当前活跃的 Agent 别名，用于多角色渲染

        Returns:
            List[Dict]: OpenAI 格式的 messages
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        return _builder.build_messages(blocks, current_agent_id=current_agent_id)

    @staticmethod
    def snapshots_to_context_text(
        snapshots: List[TopicSnapshot],
    ) -> str:
        """
        将话题快照列表转换为文本格式，供 TheEye 使用

        格式示例:
        【活跃话题列表】
        1. [T_01: 编写贪吃蛇游戏]
           状态: 已完成基础逻辑，正在调试碰撞检测
           最后对话:
           User: 把移动速度调快一点
           Assistant: 好的，我已经将移动速度从100ms调整到50ms...

        2. [T_02: 晚餐食谱推荐]
           最后对话:
           User: 推荐一个简单的晚餐食谱
           Assistant: 我推荐番茄炒蛋...

        Args:
            snapshots: 话题快照列表

        Returns:
            str: 格式化的文本
        """
        if not snapshots:
            return ""

        lines = ["【活跃话题列表】"]

        for idx, snapshot in enumerate(snapshots, 1):
            lines.append(f"{idx}. [{snapshot.topic_id}: {snapshot.title}]")

            # 添加状态摘要（如果有）
            if snapshot.state_summary:
                lines.append(f"   状态: {snapshot.state_summary}")

            # 添加最后一轮对话（如果有）
            if snapshot.last_turn:
                lines.append("   最后对话:")
                user_msg = snapshot.last_turn.get("user", "")
                assistant_msg = snapshot.last_turn.get("assistant", "")

                # 截断过长的消息
                max_len = 200
                if len(user_msg) > max_len:
                    user_msg = user_msg[:max_len] + "..."
                if len(assistant_msg) > max_len:
                    assistant_msg = assistant_msg[:max_len] + "..."

                lines.append(f"   User: {user_msg}")
                lines.append(f"   Assistant: {assistant_msg}")

            lines.append("")  # 空行分隔

        return "\n".join(lines)
