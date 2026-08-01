from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameExecutionPolicy:
    """Explicit capabilities and budget for one frame execution."""

    permitted_verbs: frozenset[str] | None = None
    max_iterations: int | None = None

    @classmethod
    def from_profile(
        cls,
        profile: object | None,
        *,
        max_iterations: int | None = None,
        denied_verbs: Iterable[str] = (),
    ) -> FrameExecutionPolicy:
        configured = getattr(profile, "allowed_mtp_verbs", None)
        permitted = None if configured is None else frozenset(v.upper() for v in configured)
        denied = {verb.upper() for verb in denied_verbs}
        if permitted is not None:
            permitted = frozenset(permitted.difference(denied))
        elif denied:
            # ``None`` means all profile verbs, so represent the denial with a
            # complete set only when the protocol's known verbs are available.
            from hivememory.core.mtp.models import MTPVerb

            permitted = frozenset(v.value for v in MTPVerb).difference(denied)
        return cls(permitted_verbs=permitted, max_iterations=max_iterations)

    def allows(self, verb: str) -> bool:
        return self.permitted_verbs is None or verb.upper() in self.permitted_verbs


__all__ = ["FrameExecutionPolicy"]
