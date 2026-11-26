"""依赖中立的引用解析结果，供 workspace 读取视图与内容编译器共同使用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom, PendingAtomSettlement
from hivememory.core.models.topic import TopicData


@dataclass
class ReferenceResolution:
    """单个引用的逐项结果；pending 与 atom 由解析器交付独立副本。

    ``topic`` 是话题句柄的读取结果：``topic`` 是读到的话题，``topic_entry`` 是句柄中的进展
    记录条目序号（从 1 数起），读索引时为空。话题句柄读不到时为 ``expired``，条目序号越界时
    为 ``not_found``。
    """

    kind: Literal[
        "pending", "redirect", "discarded", "failed", "expired", "atom", "topic", "not_found"
    ]
    requested_alias: str | None = None
    canonical_alias: str | None = None
    canonical_uuid: str | None = None
    pending: PendingAtom | None = None
    atom: MemoryAtom | None = None
    settlement: PendingAtomSettlement | None = None
    topic: TopicData | None = None
    topic_entry: int | None = None


__all__ = ["ReferenceResolution"]
