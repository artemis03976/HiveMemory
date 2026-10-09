from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameProducts:
    """从一次成功收尾的 frame 投影出的产物（仅供当前 CALL 使用）。"""

    artifact_aliases: tuple[str, ...] = ()


__all__ = ["FrameProducts"]
