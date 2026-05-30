"""
PendingAtom 生命周期状态体系（统一版）。

将散落的状态字符串与枚举收敛为三个正交维度：
- Status: 生命周期阶段（PENDING / MATERIALIZING / SETTLED / FAILED / EXPIRED / CANCELLED）
- Resolution: SETTLED 阶段下的终结分类（CREATED / MERGED / TOUCHED / UPDATED / DISCARDED）
- Kind: WRITE / UPDATE 来源（暂由别名前缀承载，本模块不直接管理）

设计依据：docs/mod/PendingAtomStatusUnificationDesign.md

迁移分阶段进行：
- Commit 1（本模块）：新增枚举与 PendingAtomSnapshot 视图，PendingAtomCache 暴露
  snapshot(alias)，旧的 runtime.models.PendingAtomStatus 与 Settlement.status 暂不动。
- Commit 2：Settlement.status -> resolution，Engine / Cache 切换为枚举。
- Commit 3：Resolver / Compiler 派生路径收敛，删除旧 enum 与 status_map。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


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


class PendingAtomSnapshot(BaseModel):
    """
    PendingAtom 的运行期视图。

    作为 PendingAtomCache 对外暴露的统一查询结构，以强类型携带
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
    """
    将旧 runtime.models.PendingAtomStatus 字符串映射到 (新 status, resolution)。

    迁移期辅助：用于 PendingAtomCache.snapshot() 的 fallback 派生路径。
    Commit 2 切换 Settlement 字段后，cache 内部直接使用新 status / resolution，
    本函数仅在 Snapshot 派生路径中使用，无需删除。
    """
    mapped = _LEGACY_STATUS_MAP.get(legacy_value)
    if mapped is None:
        raise ValueError(f"Unknown legacy PendingAtomStatus value: {legacy_value!r}")
    return mapped


__all__ = [
    "PendingAtomStatus",
    "PendingAtomResolution",
    "PendingAtomSnapshot",
    "is_legal_transition",
    "allowed_transitions",
    "map_legacy_status",
]
