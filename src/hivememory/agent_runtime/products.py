from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.models.pending import PendingAtomMaterializeTask


@dataclass(frozen=True)
class FrameProducts:
    """从一次成功收尾的 frame 投影出的产物（仅供当前 CALL 使用）。"""

    artifact_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeProducts:
    """根 run 终态后交给 Patchouli 的物化任务投影。"""

    materialize_tasks: tuple[PendingAtomMaterializeTask, ...] = ()


__all__ = ["FrameProducts", "RuntimeProducts"]
