from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameExecutionPolicy:
    """单次 frame 执行的显式能力白名单与迭代预算。"""

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
        """从 Agent Profile 白名单构造策略，并可选移除被拒绝的动词。"""
        configured = getattr(profile, "allowed_mtp_verbs", None)
        permitted = None if configured is None else frozenset(v.upper() for v in configured)
        denied = {verb.upper() for verb in denied_verbs}
        if permitted is not None:
            permitted = frozenset(permitted.difference(denied))
        elif denied:
            # ``None`` 表示全部动词均允许；只有协议已知动词集合可用时，
            # 才用完整集合来表达"除被拒绝动词外全部允许"。
            from hivememory.core.mtp.models import MTPVerb

            permitted = frozenset(v.value for v in MTPVerb).difference(denied)
        return cls(permitted_verbs=permitted, max_iterations=max_iterations)

    def allows(self, verb: str) -> bool:
        """None 表示全部允许；否则只允许白名单中的动词（大小写不敏感）。"""
        return self.permitted_verbs is None or verb.upper() in self.permitted_verbs


__all__ = ["FrameExecutionPolicy"]
