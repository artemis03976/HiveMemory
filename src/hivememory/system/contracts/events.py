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
    """Global pub/sub event names for GlobalSystemBus."""
    PENDING_ATOM_SETTLED = "alice.events.pending_atom.settled"
    PENDING_ATOM_FAILED = "alice.events.pending_atom.failed"
    PENDING_ATOM_CANCELLED = "alice.events.pending_atom.cancelled"
