"""
HiveMemory 感知层上下文转换门面

当前文件只保留轻量门面职责：
    - `blocks_to_messages()` 委托 `HistoryTranscriptBuilder`
    - `snapshots_to_context_text()` 将 topic snapshots 渲染为 TheEye 菜单文本
"""

from typing import List, Dict
from hivememory.engines.perception.models import LogicalBlock, TopicSnapshot
from hivememory.prompts.transcript import HistoryTranscriptBuilder

_history_builder = HistoryTranscriptBuilder()


class PerceptionContextConverter:
    """感知层上下文转换门面。"""

    @staticmethod
    def blocks_to_messages(
        blocks: List[LogicalBlock],
        current_agent_id: str = "default",
    ) -> List[Dict[str, str]]:
        """
        将 LogicalBlock 列表转换为 OpenAI messages 格式。

        多角色历史渲染 (Phase 1):
        当 block.identity.agent_id 与 current_agent_id 不同时，
        在 assistant 消息头部追加身份标识前缀，防止"认领幻觉"。

        Args:
            blocks: LogicalBlock 列表
            current_agent_id: 当前活跃的 Agent 别名，用于多角色渲染

        Returns:
            List[Dict]: OpenAI 格式的 messages
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        return _history_builder.build_messages(blocks, current_agent_id=current_agent_id)

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
            lines.append(f"{idx}. [{snapshot.topic_id}: {snapshot.topic_title}]")

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
