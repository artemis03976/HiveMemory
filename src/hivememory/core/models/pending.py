"""
HiveMemory 核心数据模型 - Pending / Settlement 领域

PendingAtom 由 workspace 的写入意图登记拥有，结算视图供生成引擎与登记共同使用。
状态枚举与 Focus 是供写入登记和记忆生成共同使用的跨域模型，只从本模块或
``hivememory.core.models`` 导入。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.identity import ActorIdentity, WorkspaceIdentity

# ===========================================================================
# 生命周期状态体系
# ===========================================================================


class PendingAtomStatus(str, Enum):
    """PendingAtom 生命周期阶段。

    登记即进入 MATERIALIZING：意图在 actor 操作内被提交给生成队列，之后由
    结算推进为 SETTLED、FAILED 或 CANCELLED。EXPIRED 是三者共同的永久终态；
    当前没有转入它的触发，结算后的句柄保留到进程重启。
    """

    MATERIALIZING = "materializing"
    SETTLED = "settled"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """终态：不可再迁移。EXPIRED 是唯一永久终态；其余终态仍可迁入 EXPIRED。"""
        return self == PendingAtomStatus.EXPIRED

    @property
    def is_in_flight(self) -> bool:
        """在途：已登记并提交，等待生成链结算。"""
        return self == PendingAtomStatus.MATERIALIZING


class PendingAtomResolution(str, Enum):
    """SETTLED 状态下的终结分类（其他状态此字段为 None）。"""

    CREATED = "created"  # 去重决策 = CREATE
    MERGED = "merged"  # 去重决策 = UPDATE
    TOUCHED = "touched"  # 去重决策 = TOUCH
    UPDATED = "updated"  # Mode C UPDATE 应用完成
    DISCARDED = "discarded"  # 去重决策 = DISCARD

    @property
    def has_canonical(self) -> bool:
        """该 resolution 是否会产生 canonical_uuid。"""
        return self != PendingAtomResolution.DISCARDED


_TRANSITIONS: dict[PendingAtomStatus, frozenset[PendingAtomStatus]] = {
    PendingAtomStatus.MATERIALIZING: frozenset(
        {
            PendingAtomStatus.SETTLED,
            PendingAtomStatus.FAILED,
            PendingAtomStatus.CANCELLED,
        }
    ),
    PendingAtomStatus.SETTLED: frozenset({PendingAtomStatus.EXPIRED}),
    PendingAtomStatus.FAILED: frozenset({PendingAtomStatus.EXPIRED}),
    PendingAtomStatus.EXPIRED: frozenset(),
    PendingAtomStatus.CANCELLED: frozenset({PendingAtomStatus.EXPIRED}),
}


def is_legal_transition(
    from_status: PendingAtomStatus,
    to_status: PendingAtomStatus,
) -> bool:
    """检查从 from_status 到 to_status 是否为合法迁移。"""
    return to_status in _TRANSITIONS[from_status]


# ===========================================================================
# WRITE / UPDATE 指令聚焦内容
# ===========================================================================


class WriteFocus(BaseModel):
    """WRITE 指令的 Agent 提交参数（纯 DTO，不含关联键）。

    关联键（pending_alias / intent_id / belong_to / from_actor）由 PendingAtom 持有，
    通过 PendingAtomMaterializeTask 出境，不再穿透 Focus。
    """

    content: str
    reason: str | None = None
    title: str | None = None
    model_config = ConfigDict(frozen=True)


class UpdateFocus(BaseModel):
    """UPDATE 指令的 Agent 提交参数（纯 DTO，不含关联键）。

    关联键（pending_alias / intent_id / belong_to / from_actor）由 PendingAtom 持有，
    通过 PendingAtomMaterializeTask 出境，不再穿透 Focus。
    """

    instruction: str
    content: str | None = None
    base_uuid: str
    base_alias: str
    model_config = ConfigDict(frozen=True)


# ===========================================================================
# Settlement 结算视图
# ===========================================================================


class PendingAtomSettlement(BaseModel):
    """
    Pending intent 的结算视图。

    由 GenerationEngine 在生成完成后产出，通过 GlobalSystemBus 回填到 workspace 登记。
    只有 MTP WRITE/UPDATE 触发的主动写入链路（携带 intent_id）才会生成 settlement。

    Note:
        ``resolution`` 直接以 ``PendingAtomResolution`` 强类型表达终结分类。
        ``FAILED`` 不属于 resolution（生命周期阶段而非结算分类），通过 ``error`` 字段
        与上游 ``PendingAtomStatus.FAILED`` 协同表达。
    """

    pending_alias: str
    intent_id: str
    resolution: PendingAtomResolution
    canonical_alias: str | None = None
    canonical_uuid: str | None = None
    message: str = ""
    error: str | None = None
    reason: str | None = None


# ===========================================================================
# PendingAtom 运行时句柄
# ===========================================================================


class PendingAtomMaterializeTask(BaseModel):
    """跨子系统的不可变物化请求。

    actor API 登记 WRITE/UPDATE 意图后立即提交，Patchouli 的意图提交服务从
    ``MemoryIntent`` 构造本任务并交给生成队列；finalize 不派发写入。与
    PendingAtomSettlement 构成请求/应答对偶。

    字段下游流向（不相交）：
        pending_alias / intent_id / source_verb → patchouli 组装 Settlement、分发 mode b/c
        focus   → engine._process_mode_b/c 的提取/合并
        belong_to / from_actor → 生成任务的资源归属与发起者，独立携带
    """

    pending_alias: str
    intent_id: str
    source_verb: Literal["WRITE", "UPDATE"]
    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
    focus: WriteFocus | UpdateFocus
    model_config = ConfigDict(frozen=True)


class PendingAtom(BaseModel):
    """
    运行时待物化记忆句柄。

    不是正式 MemoryAtom，不承诺最终落库。
    同一 Workspace 的 Agent 可通过 pending_alias 回读，进程关联不参与授权。
    """

    pending_alias: str
    intent_id: str  # 总是由 register_write/register_update 生成，非 Optional
    status: PendingAtomStatus
    source_verb: Literal["WRITE", "UPDATE"]

    focus: WriteFocus | UpdateFocus
    belong_to: WorkspaceIdentity
    from_actor: ActorIdentity
    process_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    # 结算跟踪
    settlement: PendingAtomSettlement | None = None


__all__ = [
    # 状态体系
    "PendingAtomStatus",
    "PendingAtomResolution",
    "is_legal_transition",
    # Focus 相关字段
    "WriteFocus",
    "UpdateFocus",
    # 主体
    "PendingAtom",
    "PendingAtomSettlement",
    "PendingAtomMaterializeTask",
]
