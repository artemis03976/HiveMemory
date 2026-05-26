"""Patchouli subsystem-local pub/sub event names."""


class PatchouliLocalEvents:
    """Events published on PatchouliBus inside the Patchouli subsystem."""

    PENDING_ATOM_SETTLED = "patchouli.events.pending_atom.settled"


__all__ = ["PatchouliLocalEvents"]
