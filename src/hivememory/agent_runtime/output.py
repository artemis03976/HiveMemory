"""单 frame 执行输出端口。

这里的输出属于当前 Agent run 的交互数据面，会参与 token streaming 和背压；
它不是全局 best-effort 的 RuntimeEvent 可观测性事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TokenDelta:
    content: str


@dataclass(frozen=True, slots=True)
class MTPStarted:
    verb: str
    target: str
    args: dict[str, Any]
    raw_text: str
    iteration: int
    action_id: str


@dataclass(frozen=True, slots=True)
class MTPFinished:
    verb: str
    target: str
    args: dict[str, Any]
    raw_text: str
    status: str
    iteration: int
    action_id: str


type FrameOutput = TokenDelta | MTPStarted | MTPFinished


class FrameOutputSink(Protocol):
    """Agent Runtime 依赖的最窄输出端口。"""

    @property
    def streams_tokens(self) -> bool: ...

    async def send(self, output: FrameOutput) -> None: ...


class NullFrameOutputSink:
    @property
    def streams_tokens(self) -> bool:
        return False

    async def send(self, output: FrameOutput) -> None:
        del output


__all__ = [
    "FrameOutput",
    "FrameOutputSink",
    "MTPFinished",
    "MTPStarted",
    "NullFrameOutputSink",
    "TokenDelta",
]
