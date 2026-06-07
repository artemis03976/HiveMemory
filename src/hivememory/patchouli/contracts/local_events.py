"""Patchouli subsystem-local pub/sub event names."""


class PatchouliLocalEvents:
    """Events published on PatchouliBus inside the Patchouli subsystem."""

    PENDING_ATOM_SETTLED = "patchouli.events.pending_atom.settled"
    PENDING_ATOM_FAILED = "patchouli.events.pending_atom.failed"
    PENDING_ATOM_CANCELLED = "patchouli.events.pending_atom.cancelled"
    MEMORY_TASK_ITEM_STATUS = "patchouli.events.memory_task.item_status"


__all__ = ["PatchouliLocalEvents"]
