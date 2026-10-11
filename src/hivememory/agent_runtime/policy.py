from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hivememory.core.mtp.models import MTPVerb


@dataclass(frozen=True)
class FrameExecutionPolicy:
    """单次 frame 可执行的 MTP 动词集合与迭代预算，不承担资源授权。"""

    # 通用 frame 默认采用协议已知动词全集；Alice 会传入本进程的明确集合。
    permitted_verbs: frozenset[str] = frozenset(verb.value for verb in MTPVerb)
    max_iterations: int | None = None

    @classmethod
    def from_verbs(
        cls,
        permitted_verbs: Iterable[str],
        *,
        max_iterations: int | None = None,
        denied_verbs: Iterable[str] = (),
    ) -> FrameExecutionPolicy:
        """从明确的动词集合构造策略，并可选移除子 frame 禁止的动词。"""
        permitted = frozenset(verb.upper() for verb in permitted_verbs)
        denied = {verb.upper() for verb in denied_verbs}
        return cls(permitted_verbs=permitted.difference(denied), max_iterations=max_iterations)

    def allows(self, verb: str) -> bool:
        """只允许集合中的动词；大小写不敏感，未知动词不会默认放行。"""
        return verb.upper() in self.permitted_verbs


__all__ = ["FrameExecutionPolicy"]
