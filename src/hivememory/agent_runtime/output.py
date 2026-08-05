"""单 frame 执行输出端口。

这里的输出属于当前 Agent run 的交互数据面，会参与 token streaming 和背压；
它不是全局 best-effort 的 RuntimeEvent 可观测性事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TokenDelta:
    """模型生成的自然语言增量（尚未检测到 MTP 左定界符）。"""

    content: str


@dataclass(frozen=True, slots=True)
class MTPStarted:
    """Runtime 已识别并准备执行一条 MTP 指令。"""

    verb: str
    target: str
    args: dict[str, Any]
    raw_text: str
    iteration: int
    action_id: str


@dataclass(frozen=True, slots=True)
class MTPFinished:
    """一条 MTP 指令执行完成后的结构化终态。"""

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
