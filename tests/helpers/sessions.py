"""会话轮次测试数据；固定写入时间，不依赖墙上时钟。"""

from datetime import UTC, datetime

from hivememory.core.protocol.models import ConversationTurn, InteractionPayload


def make_conversation_turn(*, agent_id: str = "default", **payload_fields) -> ConversationTurn:
    """用真实交互模型构建一轮历史。"""
    return ConversationTurn(
        interaction_id="history-process",
        agent_id=agent_id,
        topic_id="topic-1",
        created_at=datetime(2026, 10, 15, tzinfo=UTC),
        payload=InteractionPayload(**payload_fields),
    )
