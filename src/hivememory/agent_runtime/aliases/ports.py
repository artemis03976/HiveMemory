"""Agent runtime 侧 L1 记忆原子缓存的窄化端口。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hivememory.core.models import MemoryAtom, WorkspaceIdentity


@runtime_checkable
class AtomCachePort(Protocol):
    """Workspace 分区的 L1 记忆原子缓存端口。

    alias 的写入、读取与失效必须携带 ``WorkspaceIdentity``，Workspace 之间
    的同名 alias 互不可见；``get_atom_by_uuid`` 走全局索引（UUID 是全局资源
    ID）。缓存命中只代表存在加速对象，不替代授权——ownership 与 actor
    policy 由调用方在最终资源 owner 边界重验。
    """

    def ingest_atoms(
        self,
        atoms: list[MemoryAtom],
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """批量缓存原子并在指定 Workspace 分区内注册别名。"""
        ...

    def ingest_atom(
        self,
        atom: MemoryAtom,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """缓存单个原子并在指定 Workspace 分区内注册别名。"""
        ...

    def get_atom_by_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> MemoryAtom | None:
        """读取指定 Workspace 分区内的别名缓存，未命中返回 None。"""
        ...

    def get_atom_by_uuid(self, uuid: str) -> MemoryAtom | None:
        """通过 UUID 读取全局原子缓存，未命中返回 None。"""
        ...

    def invalidate_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """使指定 Workspace 分区内的别名及其对应原子缓存失效。"""
        ...


__all__ = ["AtomCachePort"]
