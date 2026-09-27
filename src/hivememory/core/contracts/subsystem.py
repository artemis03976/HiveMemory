from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SubsystemProtocol(Protocol):
    """子系统最小契约 — 所有子系统（Patchouli, Alice 等）必须实现此协议。"""

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> dict[str, Any]: ...
