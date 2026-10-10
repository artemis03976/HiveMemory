"""依赖中立的引用解析结果，供 workspace 读取视图与内容编译器共同使用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement


@dataclass
class ReferenceResolution:
    """单个引用的逐项结果；pending 与 atom 由解析器交付独立副本。"""

    kind: Literal["pending", "redirect", "discarded", "failed", "expired", "atom", "not_found"]
    requested_alias: str | None = None
    canonical_alias: str | None = None
    canonical_uuid: str | None = None
    pending: PendingAtom | None = None
    atom: MemoryAtom | None = None
    settlement: PendingAtomSettlement | None = None


__all__ = ["ReferenceResolution"]
