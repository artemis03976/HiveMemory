from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SystemEventType(str, Enum):
    SUBSYSTEM_STARTED = "system.subsystem.started"
    SUBSYSTEM_STOPPED = "system.subsystem.stopped"
    SYSTEM_READY = "system.ready"
    SYSTEM_SHUTTING_DOWN = "system.shutting_down"


@dataclass(frozen=True)
class SystemEvent:
    event_type: SystemEventType
    subsystem_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GlobalEvents:
    """GlobalSystemBus 的全局 pub/sub 事件名。"""

    PATCHOULI_MEMORY_CHANGED = "patchouli.events.memory.changed"
    # 写入意图的终态由 Patchouli 发布、workspace 写入意图登记订阅。settled 的
    # settlement 与 failed/cancelled 的载荷都带 pending_alias 与 intent_id，
    # 登记按两者严格匹配。
    PENDING_ATOM_SETTLED = "patchouli.events.pending_atom.settled"
    PENDING_ATOM_FAILED = "patchouli.events.pending_atom.failed"
    PENDING_ATOM_CANCELLED = "patchouli.events.pending_atom.cancelled"
