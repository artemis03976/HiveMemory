"""chat 任务进程的 CPU 输入清单契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import (
    AgentProfile,
    IdentityScope,
    MemoryAtom,
    TopicData,
)


class CPUInputManifest(BaseModel):
    """任务进程在分配 CPU 时组装的、与 CPU 无关的输入清单。

    Profile 解析、附件租借与记忆/附件编译都在进程侧完成，CPU（当前为
    Alice Agent run）只消费成品文本与未编译的检索原子，不再接触
    Patchouli prepare 的内部结构，也不再为它组装专属上下文。

    ``topic_id`` / ``topic_context`` 暂时保留：Topic 的准备与话题上下文
    在 ConversationSession 一批之前仍由 Patchouli 处理。
    """

    model_config = ConfigDict(frozen=True)

    process_id: str = Field(
        description="任务进程唯一标识；同时充当本次 Interaction 的稳定关联 ID",
    )
    identity_scope: IdentityScope = Field(description="请求级身份作用域（进程创建时冻结）")
    user_message: str = Field(description="原始用户消息")
    agent_profile: AgentProfile = Field(description="CPU 分配时经 Patchouli 公开路由解析的 Profile")
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
    topic_id: str = Field(default="", description="本轮 Topic 准备结果；暂时保留")
    topic_context: TopicData | None = Field(default=None, description="话题上下文；暂时保留")


__all__ = [
    "CPUInputManifest",
]
