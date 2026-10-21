"""
模块间通信协议模型

定义在子系统之间传递的检索请求、交互载荷与会话轮次。
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import (
    IdentityScope,
    TraceItem,
    TurnEvent,
    WorkspaceAssetRef,
)
from hivememory.core.models.interaction import InteractionOutcome
from hivememory.core.models.query import QueryFilters


class RetrievalRequest(BaseModel):
    """
    检索请求

    语义检索公共路由 ``patchouli.public.memory.retrieve`` 的输入，用于热路径的
    实时记忆检索。由授权点（能力层的 ``memory.search``、Passive 记忆上下文）
    以授权得到的 scope 构造。

    过滤语义：
    - Patchouli 在公共边界把 ``identity_scope`` 拆为资源归属与发起者：归属是
      Workspace 硬边界，可见性按发起者过滤
    - MTP SEARCH 指令可通过 filters 字段叠加额外过滤维度 (如 type:CODE)

    Attributes:
        semantic_query: 指代消解后的完整查询，用于语义检索
        keywords: 稀疏检索关键词列表（BM25）
        identity_scope: 请求者身份与 Workspace hard boundary
        filters: MTP filter 解析后的过滤条件 (可选，叠加到 identity_scope 基线之上)

    Examples:
        >>> request = RetrievalRequest(
        ...     semantic_query="如何部署贪吃蛇游戏？",
        ...     keywords=["部署", "贪吃蛇", "游戏"],
        ...     identity_scope=scope,  # 授权点返回的 scope
        ... )
    """

    # 指代消解后的完整查询（用于语义检索）
    semantic_query: str = Field(..., description="指代消解后的查询")

    # 稀疏检索关键词（用于 BM25）
    keywords: list[str] = Field(default_factory=list, description="检索关键词")

    # 请求级 Workspace hard boundary
    identity_scope: IdentityScope

    # MTP SEARCH 指令传入的过滤条件 (可选)
    filters: QueryFilters | None = Field(default=None, description="MTP filter 过滤条件")

    top_k: int | None = Field(
        default=None, ge=0, description="本次检索结果条数；缺省时使用检索引擎配置的默认条数"
    )


class InteractionPayload(BaseModel):
    """
    提交方封口、交由 Perception 消费的原子交互传输包

    作为系统级数据传输协议存在，承载单轮交互在进入感知层前的完整结构化结果。
    主动链路与 Import Bus 一致：payload 由提交方组装并封口，Patchouli finalize 原样
    提交。

    Attributes:
        user_message: 原始用户消息
        mtp_traces: 封口方从结构化轮次事件归约得到的 Trace 列表
        rewritten_query: Gateway 重写后的查询
        worth_saving: Gateway 价值判断
        assistant_final_text: 最终自然语言回复
        turn_events: 结构化轮次事件列表

    Note:
        不内嵌身份；归属与发起者由 ``InteractionSubmission`` 的
        ``belong_to``、``from_actor`` 独立承载，避免 payload 成为第二份身份事实。
    """

    outcome: InteractionOutcome = Field(default="completed", description="本轮执行结局")
    user_message: str = Field(..., description="原始用户消息")

    rewritten_query: str | None = Field(default=None, description="Gateway 重写后的查询")

    # ========== 结构化轮次事件 ==========
    # 模型最终自然语言回复
    assistant_final_text: str | None = Field(
        default=None, description="去除 MTP 噪音后的最终自然语言回复"
    )
    # 收集的结构化轮次事件列表
    turn_events: list[TurnEvent] = Field(
        default_factory=list,
        description="结构化轮次事件列表（主动链路来自 Agent 执行循环，Import Bus 来自导入的整轮记录），有值时感知层优先走结构化路径",
    )
    mtp_traces: list[TraceItem] = Field(
        default_factory=list,
        description="由封口方在提交前从结构化轮次事件归约得到的 Trace 列表",
    )

    worth_saving: bool | None = Field(default=None, description="Gateway 价值判断")
    # 本次 run 实际使用的模型展示名（来自 CPU 执行结果的 model_used）
    # 写入短期话题快照，供 TopicSnapshot 展示给前端
    model_used: str = Field(
        default="", description="实际使用的模型展示名，空字符串表示注册表未启用"
    )

    # 实际进入 attachment_context 的使用集合（唯一附件事实）：handler
    # 据此一次性投影 bound refs 建立 binding。Import Bus 提交没有附件，投影为空数组。
    used_attachments: list[WorkspaceAssetRef] = Field(
        default_factory=list,
        description="实际进入本轮上下文的附件使用引用快照；Import Bus 提交投影为空数组",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class SessionCompactionCheckpoint(BaseModel):
    """已封口历史的派生视图；游标引用轮次，正文仍由会话原文持有。"""

    producer: str
    format_version: int = Field(ge=1)
    reduced_through: str | None = None
    summarized_through: str | None = None
    summary: str = ""
    references: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(frozen=True)


class ConversationTurn(BaseModel):
    """一轮已封口的交互，正文只保存在 payload 中。"""

    interaction_id: str
    agent_id: str
    payload: InteractionPayload
    topic_id: str
    created_at: datetime

    model_config = ConfigDict(frozen=True)


__all__ = [
    "ConversationTurn",
    "SessionCompactionCheckpoint",
    "QueryFilters",
    "RetrievalRequest",
    "InteractionPayload",
]
