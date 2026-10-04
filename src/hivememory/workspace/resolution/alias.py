"""workspace memory read 能力：canonical 引用的多级解析与交付授权（A2 §2.2）。

迁移源为 Alice 侧 ``agent_runtime/aliases/resolver.py``（``RuntimeAliasResolver``）
的 L1/L2 部分；按边界宪章 §6.1 归 workspace runtime。解析顺序：

- L0 pending registry：A4 接入后启用（接缝形状由 A4-0 冻结），缺省等价 L0 miss；
- L1 ``AtomCache``：按原子自身 policy 对当前 Actor 授权，有效命中不回源；
- L2 backing 冷读：受 Workspace 代次守护，代次未变才回填，库侧资源归属与
  policy 校验独立成立（纵深防御）。

resolver 不做检索排序、内容解释或任何写入；只缓存 Workspace 共享的完整
原子，不缓存 Actor 不可见/缺失结果，也不缓存任何 Actor 的授权结论。
"""

from __future__ import annotations

import logging
from uuid import UUID

from hivememory.core.memory_access import memory_belongs_to_workspace, memory_is_readable
from hivememory.core.models import IdentityScope, MemoryAtom
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.workspace.cache.atom import AtomCache
from hivememory.workspace.resolution.backing import CanonicalReadBackend
from hivememory.workspace.resolution.guard import ColdReadGuard

logger = logging.getLogger(__name__)


class AliasResolver:
    """canonical alias/UUID 的 L0/L1/L2 解析器，交付完整原子独立副本。"""

    def __init__(
        self,
        *,
        cache: AtomCache,
        guard: ColdReadGuard,
        backing: CanonicalReadBackend,
    ) -> None:
        self._cache = cache
        self._guard = guard
        self._backing = backing

    async def read(
        self,
        memory_id: UUID,
        *,
        scope: IdentityScope,
    ) -> MemoryAtom | None:
        """UUID 点读：未知或对当前 Actor 不可见时返回 None（A1 防泄露）。"""
        self._guard.ensure_open()
        workspace = scope.workspace_identity
        cached = self._cache.get_by_id(workspace, memory_id)
        if cached is not None:
            return cached if self._readable(cached, scope) else None

        atom, fill = await self._guard.load(
            workspace,
            lambda: self._backing.read(memory_id, scope=scope),
        )
        if atom is None:
            # 缺失与不可见被合并为 None，不能据此写共享负缓存（A2 §3.2）。
            return None
        return self._admit(atom, scope, fill=fill)

    async def resolve_aliases(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
    ) -> list[MemoryAtom]:
        """alias 批量读取：结果按请求顺序，只包含实际可读的完整原子。

        alias 先去首尾空白并按首次出现去重；缺失或不可见的 alias 不出现在
        结果中，不以列表下标表达逐项状态。
        """
        self._guard.ensure_open()
        workspace = scope.workspace_identity
        requested = _normalize_aliases(aliases)
        delivered: dict[str, MemoryAtom] = {}
        misses: list[str] = []
        for alias in requested:
            # L0：pending registry 分支，A4 接入前恒为 miss。
            cached = self._cache.get_by_alias(workspace, alias)
            if cached is None:
                misses.append(alias)
            elif self._readable(cached, scope):
                delivered[alias] = cached

        if misses:
            atoms, fill = await self._guard.load(
                workspace,
                lambda: self._backing.retrieve_by_aliases(misses, scope=scope),
            )
            pending = set(misses)
            for atom in atoms:
                alias = atom.index.alias
                if alias not in pending:
                    continue
                admitted = self._admit(atom, scope, fill=fill)
                if admitted is not None:
                    delivered[alias] = admitted
        return [delivered[alias] for alias in requested if alias in delivered]

    async def search(
        self,
        request: RetrievalRequest,
        *,
        scope: IdentityScope,
    ) -> list[MemoryAtom]:
        """语义检索：仍执行检索并保持领域排序，结果协作预热 AtomCache。

        缓存不替代语义搜索；检索结果直接来自存储，读取期间代次变化时照常
        返回但不预热，避免把并发变更前的值写入共享缓存。
        """
        self._guard.ensure_open()
        atoms, fill = await self._guard.load(
            scope.workspace_identity,
            lambda: self._backing.retrieve(request),
            retry_on_stale=False,
        )
        delivered = []
        for atom in atoms:
            admitted = self._admit(atom, scope, fill=fill)
            if admitted is not None:
                delivered.append(admitted)
        return delivered

    def _admit(self, atom: MemoryAtom, scope: IdentityScope, *, fill: bool) -> MemoryAtom | None:
        """冷读结果的归属验证、回填与交付授权。"""
        workspace = scope.workspace_identity
        if not memory_belongs_to_workspace(atom, workspace):
            # 纵深防御：backing 返回越界原子时既不交付也不回填，不扩大权限。
            logger.warning(
                "backing 返回了不属于请求 Workspace 的原子，已丢弃: memory_id=%s",
                atom.id,
            )
            return None
        if fill:
            self._cache.put(atom)
        if not self._readable(atom, scope):
            return None
        return atom.model_copy(deep=True)

    @staticmethod
    def _readable(atom: MemoryAtom, scope: IdentityScope) -> bool:
        """交付边界逐次授权：ownership 硬边界 + 原子自身 policy。"""
        return memory_is_readable(
            atom,
            workspace_identity=scope.workspace_identity,
            actor_identity=scope.actor_identity,
        )


def _normalize_aliases(aliases: list[str]) -> list[str]:
    """去首尾空白、丢弃空 alias，并按首次出现去重（沿用 backing 的既有语义）。"""
    normalized: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        value = alias.strip() if alias else ""
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


__all__ = ["AliasResolver"]
