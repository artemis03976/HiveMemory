"""Gateway topic snapshot 渲染工具。"""

from __future__ import annotations

from collections.abc import Sequence

from hivememory.core.models import TopicSnapshot


def render_topic_snapshots(snapshots: Sequence[TopicSnapshot]) -> str:
    """将活跃话题快照渲染为 Gateway 决策可读的菜单文本。"""

    if not snapshots:
        return ""

    lines = ["【活跃话题列表】"]
    for idx, snapshot in enumerate(snapshots, 1):
        lines.append(f"{idx}. [{snapshot.topic_id}: {snapshot.topic_title}]")

        if snapshot.state_summary:
            lines.append(f"   状态: {snapshot.state_summary}")

        if snapshot.last_turn:
            lines.append("   最后对话:")
            user_msg = snapshot.last_turn.get("user", "")
            assistant_msg = snapshot.last_turn.get("assistant", "")

            max_len = 200
            if len(user_msg) > max_len:
                user_msg = user_msg[:max_len] + "..."
            if len(assistant_msg) > max_len:
                assistant_msg = assistant_msg[:max_len] + "..."

            lines.append(f"   User: {user_msg}")
            lines.append(f"   Assistant: {assistant_msg}")

        lines.append("")

    return "\n".join(lines)


__all__ = ["render_topic_snapshots"]
