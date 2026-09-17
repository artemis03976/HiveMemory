"""
Workspace 资源平面的不可变快照与跨边界 DTO。

资源端口只返回本模块定义的快照/投影，不把可变的 ``MemoryAtom``、
``AgentProfile`` 或 Patchouli/Alice 私有对象泄漏给调用方（父计划 5.7.2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import (
    AgentProfile,
    IdentityScope,
    MemoryAccessPolicy,
    MemoryAtom,
    WorkspaceIdentity,
)
from hivememory.workspace.access import WorkspaceAccessContext


class MemorySnapshot(BaseModel):
    """一次授权 Memory 读取的不可变快照。

    携带 canonical identity（``memory_id`` + ``workspace_identity``）、
    source revision（``meta.version``，乐观锁版本）与内容投影；缓存命中
    的 freshness 校验（WRX-3）以 ``source_revision``/``updated_at`` 为身份
    基准。快照与 canonical atom 解耦：后续 canonical 变化不影响已取出的
    快照，也不表示快照仍然新鲜。
    """

    memory_id: str = Field(description="canonical Memory UUID（全局寻址 ID）")
    workspace_identity: WorkspaceIdentity = Field(description="资源归属 Workspace")
    alias: str = Field(description="Workspace 分区内的正式别名")
    memory_type: str = Field(description="MemoryType 值")
    title: str | None = None
    summary: str | None = None
    content: str = Field(description="payload 内容全文")
    tags: tuple[str, ...] = ()
    source_agent_id: str = Field(description="来源 provenance（非授权字段）")
    visibility: str = Field(description="MemoryAccessPolicy 可见性值")
    confidence_score: float = 0.0
    created_at: datetime
    updated_at: datetime
    source_revision: int = Field(description="canonical source 版本（meta.version）")

    @classmethod
    def from_atom(cls, atom: MemoryAtom) -> MemorySnapshot:
        """从已通过授权校验的 canonical atom 构建快照投影。"""
        policy: MemoryAccessPolicy = atom.meta.access_policy
        index = atom.index
        return cls(
            memory_id=str(atom.id),
            workspace_identity=atom.meta.workspace_identity,
            alias=atom.get_alias(),
            memory_type=index.memory_type.value,
            title=index.title,
            summary=index.summary,
            content=atom.payload.content,
            tags=tuple(index.tags or ()),
            source_agent_id=atom.meta.source_agent_id,
            visibility=policy.visibility.value,
            confidence_score=atom.meta.confidence_score,
            created_at=atom.meta.created_at,
            updated_at=atom.meta.updated_at,
            source_revision=atom.meta.version,
        )

    model_config = ConfigDict(frozen=True)


class ProfileSnapshot(BaseModel):
    """一次授权 Agent Profile 读取的快照投影。

    ``profile`` 是 canonical ``AgentProfile`` 的 copy-on-read 深拷贝：服务
    每次读取都返回独立副本，调用方修改不会影响缓存或其他调用方；快照
    本身冻结。内建 profile（default/omni_doll）以 ``source_kind="builtin"``
    显式标识，不把常量 alias 当作跨 Workspace 授权旁路。
    """

    agent_alias: str | None = Field(default=None, description="请求的正式 alias；内建为 None")
    profile: AgentProfile = Field(description="Profile 投影（copy-on-read 深拷贝）")
    source_kind: Literal["builtin", "atom"]
    source_atom_uuid: str | None = Field(default=None, description="来源 atom UUID")
    source_revision: int | None = Field(default=None, description="来源 atom 版本")

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# 领域提交 / 结果 / 失效 DTO（父计划 5.7.1 端口参数）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainHandle:
    """可定位权威领域结果的句柄（当前为 Patchouli 生成任务 ID）。"""

    task_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id 不能为空")


@dataclass(frozen=True)
class DomainSubmission:
    """意图提交回执，说明四个事实中的"是否被接纳"。

    ``accepted=False`` 表示生成 admission 未接纳（确定性拒绝或批量响应
    缺失）；接纳不等于已应用——应用进度经 ``DomainResultPort`` 以同一
    handle 查询。不返回无法定位权威结果的布尔成功值。
    """

    accepted: bool
    handle: DomainHandle | None = None
    task_status: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DomainResult:
    """领域任务的真实结果投影（对应 Patchouli ``MemoryGenerationTask``）。

    ``identity_scope``/``submitted_by`` 是 WRX-1 补齐的归属字段：查询侧
    用它们做跨 scope 授权判断，缺少归属的结果不得返回给越权查询。
    """

    task_id: str
    status: str
    canonical_alias: str | None = None
    error: str | None = None
    topic_id: str | None = None
    pending_alias: str | None = None
    identity_scope: IdentityScope | None = None
    submitted_by: str | None = None


@dataclass(frozen=True)
class MemoryIntentRequest:
    """中立的记忆意图请求：由 Patchouli 决定生成、更新、合并或丢弃。

    ``kind="write"`` 需要 ``content``；``kind="update"`` 需要 ``instruction``
    以及 ``base_alias`` + ``base_uuid``（UPDATE 的业务前提是调用方已持有
    当前 base 的坐标）。``topic_id`` 是会话锚定（生成上下文来源），必须
    显式给出。

    ``intent_id`` 是幂等键，缺省生成：同一 ``intent_id`` 携带相同载荷重试
    会命中 controller 幂等复用并返回原任务句柄；载荷变化则按 intent 冲突
    确定性拒绝（at-most-one-canonical，不产生第二个 canonical Memory）。
    """

    access: WorkspaceAccessContext
    kind: Literal["write", "update"]
    topic_id: str
    content: str | None = None
    title: str | None = None
    reason: str | None = None
    instruction: str | None = None
    base_alias: str | None = None
    base_uuid: str | None = None
    intent_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "write":
            if not self.content:
                raise ValueError("write 意图需要 content")
            if self.instruction or self.base_alias or self.base_uuid:
                raise ValueError("write 意图不得携带 update 字段")
        elif self.kind == "update":
            if not self.instruction:
                raise ValueError("update 意图需要 instruction")
            if not self.base_alias or not self.base_uuid:
                raise ValueError("update 意图需要 base_alias 与 base_uuid")
        else:
            raise ValueError(f"不支持的意图类型: {self.kind!r}")
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("topic_id 不能为空")


@dataclass(frozen=True)
class InteractionApplyRequest:
    """已发生交互的领域提交请求（承接 ``InteractionSubmissionQueue`` 语义）。

    ``payload`` 使用 core.protocol 的共享 ``InteractionPayload`` 模型；本
    模块以 object 注解避免 workspace 包对协议层的结构化耦合。
    """

    access: WorkspaceAccessContext
    payload: object
    requested_topic_id: str = "NEW_TOPIC"
    interaction_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_topic_id, str) or not self.requested_topic_id.strip():
            raise ValueError("requested_topic_id 不能为空")


@dataclass(frozen=True)
class InteractionApplyResult:
    """交互提交收据投影：请求已被队列接纳，应用进度按 interaction_id 跟踪。"""

    interaction_id: str
    work_id: str
    state: str


@dataclass(frozen=True)
class CanonicalResourceChange:
    """一次 canonical mutation 的失效通知描述（``ResourceInvalidationPort`` 参数）。

    ``kind`` 覆盖父计划 6.2 节的失效来源；派生 cache 实现据此精确失效，
    不得把通知解释为"全量清空"或授权证明。
    """

    kind: Literal[
        "memory_created",
        "memory_updated",
        "memory_deleted",
        "memory_archived",
        "memory_revived",
        "profile_changed",
        "asset_representation_changed",
    ]
    workspace_identity: WorkspaceIdentity
    memory_id: str | None = None
    alias: str | None = None
    source_revision: int | None = None


__all__ = [
    "MemorySnapshot",
    "ProfileSnapshot",
    "DomainHandle",
    "DomainSubmission",
    "DomainResult",
    "MemoryIntentRequest",
    "InteractionApplyRequest",
    "InteractionApplyResult",
    "CanonicalResourceChange",
]
