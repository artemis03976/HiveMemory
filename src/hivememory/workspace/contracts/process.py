"""chat 任务进程的 CPU 输入清单契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.access import WorkspaceOperation
from hivememory.core.models import (
    AgentProfile,
    ExecutionLabels,
    MemoryAtom,
)
from hivememory.core.protocol.models import ConversationTurn, SessionCompactionCheckpoint


class CPUInputManifest(BaseModel):
    """任务进程在分配 CPU 时组装的、与 CPU 无关的输入清单。

    Profile 解析、附件租借与记忆/附件编译都在进程侧完成，CPU（当前为
    Alice Agent run）只消费成品文本与未编译的检索原子，不再接触
    Patchouli prepare 的内部结构，也不再为它组装专属上下文。

    ``topic_id`` 只用于根帧挂载与观测；历史来自当前会话，Topic 不提供对话上下文。
    """

    model_config = ConfigDict(frozen=True)

    process_id: str = Field(
        description="任务进程唯一标识；同时充当本次 Interaction 的稳定关联 ID",
    )
    labels: ExecutionLabels = Field(description="注册时绑定的只读观测标签，不用于行为授权")
    user_message: str = Field(description="原始用户消息")
    agent_profile: AgentProfile = Field(description="CPU 分配时经阶段授权与读取视图解析的 Profile")
    allowed_operations: frozenset[WorkspaceOperation] = Field(
        description="CPU 分配时的 actor 操作快照，只供提示词与执行策略使用，不作为授权依据",
    )
    memories: list[MemoryAtom] = Field(
        default_factory=list,
        description="未编译的检索结果原子（prepare 返回的原始列表）",
    )
    memory_context: str = Field(
        default="",
        description="进程编译的 RETRIEVAL_CONTEXT 文本；检索为空时为空字符串",
    )
    attachment_context: str = Field(
        default="",
        description="进程编译的附件 section 文本；未选择附件时为空字符串",
    )
    storage_available: bool = Field(default=True, description="记忆存储健康状态")
    topic_id: str = Field(default="", description="本轮 Topic 准备结果，只用于根帧挂载与观测")
    session_history: tuple[ConversationTurn, ...] = Field(
        default_factory=tuple, description="当前会话的全量历史快照，不按 Agent 过滤或截断"
    )
    compaction_checkpoint: SessionCompactionCheckpoint | None = Field(
        default=None, description="会话持有的历史压缩检查点，由 CPU 判断格式是否可用"
    )


__all__ = [
    "CPUInputManifest",
]
