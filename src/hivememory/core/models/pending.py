"""
HiveMemory 核心数据模型 - Pending / Settlement 领域

PendingAtom 与其结算视图 PendingAtomSettlement 是 ``engines/`` 与 ``alice/`` 之间
的跨域共享物。把它们及其周边类型（状态枚举、Focus、RuntimeScope）统一上移到 core，
消除 ``engines → alice`` / ``alice → engines`` 的子系统层级倒挂。

迁移依据: docs/mod/PendingAtomRuntimeDesign.md §6.2

新代码应直接从本模块或 ``hivememory.core.models`` 导入。生成域 facade
``hivememory.engines.generation.models`` 仍 re-export 一份 PendingAtomSettlement /
DuplicateDecision / WriteFocus / UpdateFocus，作为生成流水线域内的稳定面，
调用方按业务直觉选择即可。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.models.interaction import Identity


# ===========================================================================
# 生命周期状态体系
# ===========================================================================


class PendingAtomStatus(str, Enum):
    """PendingAtom 生命周期阶段。"""

    PENDING = "pending"
    MATERIALIZING = "materializing"
    SETTLED = "settled"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """终态：不可再迁移。"""
        return self in {
            PendingAtomStatus.SETTLED,
            PendingAtomStatus.FAILED,
            PendingAtomStatus.EXPIRED,
            PendingAtomStatus.CANCELLED,
        }

    @property
    def is_in_flight(self) -> bool:
        """非终态：仍在生命周期内流转。"""
        return self in {
            PendingAtomStatus.PENDING,
            PendingAtomStatus.MATERIALIZING,
        }


class PendingAtomResolution(str, Enum):
    """SETTLED 状态下的终结分类（其他状态此字段为 None）。"""

    CREATED = "created"      # dedup decision = CREATE
    MERGED = "merged"        # dedup decision = UPDATE
    TOUCHED = "touched"      # dedup decision = TOUCH
    UPDATED = "updated"      # Mode C UPDATE 应用完成
    DISCARDED = "discarded"  # dedup decision = DISCARD

    @property
    def has_canonical(self) -> bool:
        """该 resolution 是否会产生 canonical_uuid。"""
        return self != PendingAtomResolution.DISCARDED


_TRANSITIONS: dict[PendingAtomStatus, frozenset[PendingAtomStatus]] = {
    PendingAtomStatus.PENDING: frozenset({
        PendingAtomStatus.MATERIALIZING,
        PendingAtomStatus.EXPIRED,
        PendingAtomStatus.CANCELLED,
    }),
    PendingAtomStatus.MATERIALIZING: frozenset({
        PendingAtomStatus.SETTLED,
        PendingAtomStatus.FAILED,
        PendingAtomStatus.CANCELLED,
    }),
    PendingAtomStatus.SETTLED: frozenset(),
    PendingAtomStatus.FAILED: frozenset(),
    PendingAtomStatus.EXPIRED: frozenset(),
    PendingAtomStatus.CANCELLED: frozenset(),
}


def is_legal_transition(
    from_status: PendingAtomStatus,
    to_status: PendingAtomStatus,
) -> bool:
    """检查从 from_status 到 to_status 是否为合法迁移。"""
    return to_status in _TRANSITIONS[from_status]


def allowed_transitions(from_status: PendingAtomStatus) -> frozenset[PendingAtomStatus]:
    """返回 from_status 的所有合法目标状态。"""
    return _TRANSITIONS[from_status]


_LEGACY_STATUS_MAP: dict[
    str, tuple[PendingAtomStatus, PendingAtomResolution | None]
] = {
    "pending":   (PendingAtomStatus.PENDING, None),
    "revision":  (PendingAtomStatus.PENDING, None),
    "committed": (PendingAtomStatus.SETTLED, PendingAtomResolution.CREATED),
    "merged":    (PendingAtomStatus.SETTLED, PendingAtomResolution.MERGED),
    "updated":   (PendingAtomStatus.SETTLED, PendingAtomResolution.UPDATED),
    "touched":   (PendingAtomStatus.SETTLED, PendingAtomResolution.TOUCHED),
    "discarded": (PendingAtomStatus.SETTLED, PendingAtomResolution.DISCARDED),
    "failed":    (PendingAtomStatus.FAILED, None),
}


def map_legacy_status(
    legacy_value: str,
) -> tuple[PendingAtomStatus, PendingAtomResolution | None]:
    """旧 status 字符串 → 新 (status, resolution) 元组的兼容映射。

    保留作为外部 / 历史持久化数据的反序列化入口；运行期 PendingAtomRuntime 不再使用。
    """
    mapped = _LEGACY_STATUS_MAP.get(legacy_value)
    if mapped is None:
        raise ValueError(f"Unknown legacy PendingAtomStatus value: {legacy_value!r}")
    return mapped


# ===========================================================================
# 查重决策
# ===========================================================================

# TODO:放回generation引擎
class DuplicateDecision(str, Enum):
    """
    查重决策类型

    Attributes:
        CREATE: 创建新记忆
        UPDATE: 更新现有记忆（知识演化）
        TOUCH: 仅更新访问时间（完全重复）
        DISCARD: 丢弃（低质量重复）
    """
    CREATE = "create"
    UPDATE = "update"
    TOUCH = "touch"
    DISCARD = "discard"


# ===========================================================================
# WRITE / UPDATE 指令聚焦内容
# ===========================================================================


class WriteFocus(BaseModel):
    """WRITE 指令的 Agent 提交参数（纯 DTO，不含关联键）。

    关联键（pending_alias / intent_id / identity）由 PendingAtom 持有，
    通过 PendingAtomMaterializeTask 出境，不再穿透 Focus。
    """
    content: str
    reason: Optional[str] = None
    title: Optional[str] = None
    model_config = ConfigDict(frozen=True)


class UpdateFocus(BaseModel):
    """UPDATE 指令的 Agent 提交参数（纯 DTO，不含关联键）。

    关联键（pending_alias / intent_id / identity）由 PendingAtom 持有，
    通过 PendingAtomMaterializeTask 出境，不再穿透 Focus。
    """
    instruction: str
    content: Optional[str] = None
    base_uuid: str
    base_alias: str
    model_config = ConfigDict(frozen=True)


# ===========================================================================
# 执行坐标
# ===========================================================================


class RuntimeScope(BaseModel):
    """Runtime execution coordinates for an Alice agent run."""

    run_id: str = ""
    frame_id: str = ""
    parent_frame_id: Optional[str] = None
    action_id: Optional[str] = None
    depth: int = 0

    def with_action(self, action_id: str) -> "RuntimeScope":
        """Return a copy scoped to one agent action."""
        return self.model_copy(update={"action_id": action_id})

    def for_child(self, frame_id: str) -> "RuntimeScope":
        """Return a child frame scope under the same run."""
        return RuntimeScope(
            run_id=self.run_id,
            frame_id=frame_id,
            parent_frame_id=self.frame_id,
            depth=self.depth + 1,
        )

    model_config = ConfigDict(frozen=True)


# ===========================================================================
# Settlement 结算视图
# ===========================================================================


class PendingAtomSettlement(BaseModel):
    """
    Pending intent 的结算视图。

    由 GenerationEngine 在生成完成后产出，通过 GlobalSystemBus 回填到 AliceRuntime。
    只有 MTP WRITE/UPDATE 触发的主动写入链路（携带 intent_id）才会生成 settlement。

    Note:
        ``resolution`` 直接以 ``PendingAtomResolution`` 强类型表达终结分类。
        ``FAILED`` 不属于 resolution（生命周期阶段而非结算分类），通过 ``error`` 字段
        与上游 ``PendingAtomStatus.FAILED`` 协同表达。
    """
    pending_alias: str
    intent_id: str
    resolution: PendingAtomResolution
    duplicate_decision: Optional[DuplicateDecision] = None
    canonical_alias: Optional[str] = None
    canonical_uuid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reason: Optional[str] = None


# ===========================================================================
# PendingAtom 运行时句柄
# ===========================================================================


class PendingAtomMaterializeTask(BaseModel):
    """跨子系统的不可变物化请求。

    Alice 编排层在 run 结束时从 PendingAtomRuntime 投影产出，进入 finalize 后
    patchouli 解析。与 PendingAtomSettlement 构成请求/应答对偶。

    字段下游流向（不相交）：
        pending_alias / intent_id / source_verb → patchouli 组装 Settlement、分发 mode b/c
        focus   → engine._process_mode_b/c 的提取/合并
        identity → GenerationRequest.identity
    """

    pending_alias: str
    intent_id: str
    source_verb: Literal["WRITE", "UPDATE"]
    identity: "Identity"
    focus: "WriteFocus | UpdateFocus"
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_pending_atom(cls, pa: "PendingAtom") -> "PendingAtomMaterializeTask":
        return cls(
            pending_alias=pa.pending_alias,
            intent_id=pa.intent_id,
            source_verb=pa.source_verb,
            identity=pa.identity,
            focus=pa.focus,
        )


class PendingAtom(BaseModel):
    """
    运行时待物化记忆句柄。

    不是正式 MemoryAtom，不承诺最终落库。
    在其生命周期内，Agent 可通过 pending_alias 读取本次写入意图的内容。
    """

    pending_alias: str
    intent_id: str  # 总是由 register_write/register_update 生成，非 Optional
    status: PendingAtomStatus
    source_verb: Literal["WRITE", "UPDATE"]

    focus: WriteFocus | UpdateFocus
    identity: Identity = Field(default_factory=Identity)
    runtime_scope: RuntimeScope = Field(default_factory=RuntimeScope)
    created_at: datetime = Field(default_factory=datetime.now)

    # Phase 2: settlement tracking
    settlement: Optional[PendingAtomSettlement] = None


# ===========================================================================
# PendingAtom 运行期视图
# ===========================================================================


class PendingAtomSnapshot(BaseModel):
    """
    PendingAtom 的运行期视图。

    作为 PendingAtomRuntime 对外暴露的统一查询结构，以强类型携带
    (status, resolution, canonical_*) 四元组。所有外部消费者
    （compiler / resolver / 视图层）都应以此为输入，而不是自己解析旧字符串。

    不变量：
    - status != SETTLED 时 resolution 必须为 None
    - status == SETTLED 时 resolution 必须非空
    - resolution.has_canonical 时 canonical_uuid 必须非空
    - resolution == DISCARDED 时 canonical_uuid / canonical_alias 必须为空
    """

    pending_alias: str
    status: PendingAtomStatus
    resolution: PendingAtomResolution | None = None
    canonical_alias: str | None = None
    canonical_uuid: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> PendingAtomSnapshot:
        if self.status == PendingAtomStatus.SETTLED:
            if self.resolution is None:
                raise ValueError("SETTLED status requires a resolution")
        else:
            if self.resolution is not None:
                raise ValueError(
                    f"status={self.status.value} must not carry a resolution"
                )

        if self.resolution is not None:
            if self.resolution.has_canonical:
                if self.canonical_uuid is None:
                    raise ValueError(
                        f"resolution={self.resolution.value} requires canonical_uuid"
                    )
            else:
                if self.canonical_uuid is not None or self.canonical_alias is not None:
                    raise ValueError(
                        "DISCARDED resolution must not carry canonical refs"
                    )
        return self

    model_config = ConfigDict(frozen=True)


__all__ = [
    # 状态体系
    "PendingAtomStatus",
    "PendingAtomResolution",
    "PendingAtomSnapshot",
    "is_legal_transition",
    "allowed_transitions",
    "map_legacy_status",
    # 查重决策
    "DuplicateDecision",
    # Focus
    "WriteFocus",
    "UpdateFocus",
    # 执行坐标
    "RuntimeScope",
    # 主体
    "PendingAtom",
    "PendingAtomSettlement",
    "PendingAtomMaterializeTask",
]
