"""Patchouli runtime package."""

from hivememory.patchouli.runtime.bus import PatchouliBus

__all__ = ["PatchouliBus", "PatchouliRuntime"]


def __getattr__(name: str):
    if name == "PatchouliRuntime":
        from hivememory.patchouli.runtime.core import PatchouliRuntime

        return PatchouliRuntime
    raise AttributeError(name)
