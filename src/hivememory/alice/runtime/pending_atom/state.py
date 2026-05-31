"""
[兼容入口] PendingAtom 状态体系已上移至 ``core/models/pending.py``。

PR2 子动作 A 将状态枚举、Snapshot、状态机纯函数统一上移到 core，消除
``engines → alice`` 的层级倒挂。本模块保留为旧导入路径的 re-export 兼容层，
新代码请直接从 ``hivememory.core.models`` 导入。

迁移依据: docs/mod/PendingAtomRuntimeDesign.md §6.2
"""

from hivememory.core.models.pending import (
    PendingAtomStatus,
    PendingAtomResolution,
    PendingAtomSnapshot,
    is_legal_transition,
    allowed_transitions,
    map_legacy_status,
)

__all__ = [
    "PendingAtomStatus",
    "PendingAtomResolution",
    "PendingAtomSnapshot",
    "is_legal_transition",
    "allowed_transitions",
    "map_legacy_status",
]
