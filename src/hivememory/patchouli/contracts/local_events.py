"""Patchouli 子系统内部 pub/sub 事件名。"""


class PatchouliLocalEvents:
    """Patchouli 子系统内发布到 PatchouliBus 的事件。"""

    PENDING_ATOM_SETTLED = "patchouli.events.pending_atom.settled"
    PENDING_ATOM_FAILED = "patchouli.events.pending_atom.failed"
    PENDING_ATOM_CANCELLED = "patchouli.events.pending_atom.cancelled"


__all__ = ["PatchouliLocalEvents"]
