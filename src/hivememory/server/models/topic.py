"""Topic 相关的 Response 模型"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TopicSnapshotResponse(BaseModel):
    topic_id: str
    topic_title: str
    state_summary: str = ""
    last_turn: Optional[Dict[str, str]] = None
    total_tokens: int = 0


class TopicListResponse(BaseModel):
    topics: List[TopicSnapshotResponse]


class TriggerResponse(BaseModel):
    success: bool
    topic_id: Optional[str] = None
    message: str = ""
    blocks_archived: Optional[int] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str = ""
