"""会话历史的 CPU 中立值模型；agent 标签只用于展示，不参与授权。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from hivememory.core.protocol.models import InteractionPayload


class ConversationTurn(BaseModel):
    """一轮已封口的交互，正文只保存在 payload 中。"""

    interaction_id: str
    agent_id: str
    payload: InteractionPayload
    topic_id: str
    created_at: datetime

    model_config = ConfigDict(frozen=True)
