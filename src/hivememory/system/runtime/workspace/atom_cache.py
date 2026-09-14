"""
Workspace 分区的记忆原子缓存实现（KoakumaAtomCache）。

由 WorkspaceRuntime 创建并持有所有权，Alice/Agent runtime 侧只经
``AtomCachePort``（见 ports.py）消费。提供：
- 完整 MemoryAtom 对象缓存
- 双索引：(WorkspaceIdentity, alias) 分区别名索引 + 全局 UUID -> MemoryAtom
- 会话级生命周期（无需 LRU 淘汰），附带可观测的 alias 命中/未命中统计
- UPDATE 后缓存失效支持

作者: HiveMemory Team
版本: 2.0
"""

import logging
from typing import Dict, List, Optional

from hivememory.core.models import MemoryAtom, WorkspaceIdentity

logger = logging.getLogger(__name__)


class KoakumaAtomCache:
    """
    Workspace 分区的统一记忆原子缓存，带别名解析功能

    会话级缓存，存储完整的 MemoryAtom 对象。
    消除 SEARCH/READ/RUN 流程中的冗余数据库查询。

    alias 索引按 ``(WorkspaceIdentity, alias)`` 分区，Workspace 之间的同名
    alias 互不可见；UUID 是全局资源 ID，``UUID -> MemoryAtom`` 索引保持全局。
    缓存命中只代表存在加速对象，ownership 与 actor policy 仍由 resolver 在
    资源 owner 边界重验，缓存 key 不替代授权。
    """

    def __init__(self):
        """初始化双索引缓存结构。"""
        # 核心缓存：UUID -> MemoryAtom（UUID 全局唯一，不按 Workspace 分区）
        self._uuid_to_atom: Dict[str, MemoryAtom] = {}
        # 别名映射：(WorkspaceIdentity, alias) -> UUID
        self._alias_to_uuid: Dict[tuple[WorkspaceIdentity, str], str] = {}
        # 可观测统计：alias 读取路径的命中/未命中次数
        self._alias_hits = 0
        self._alias_misses = 0

    @staticmethod
    def _require_workspace_identity(
        workspace_identity: WorkspaceIdentity,
    ) -> WorkspaceIdentity:
        """拒绝缺失或非 WorkspaceIdentity 的坐标，不提供无 scope 的公共读写。"""
        if not isinstance(workspace_identity, WorkspaceIdentity):
            raise TypeError("workspace_identity 必须是 WorkspaceIdentity")
        return workspace_identity

    def ingest_atoms(
        self,
        atoms: List[MemoryAtom],
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """
        批量缓存原子并在指定 Workspace 分区内注册别名。

        用于 SEARCH 和预检索结果的批量注册。
        """
        workspace = self._require_workspace_identity(workspace_identity)
        for atom in atoms:
            self.ingest_atom(atom, workspace_identity=workspace)

    def ingest_atom(
        self,
        atom: MemoryAtom,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """缓存单个原子并在指定 Workspace 分区内注册别名。"""
        workspace = self._require_workspace_identity(workspace_identity)
        uuid_str = str(atom.id)
        alias = atom.get_alias()
        self._uuid_to_atom[uuid_str] = atom
        # 同一 (Workspace, alias) 重复写入按替换语义覆盖，不会留下两个相互
        # 矛盾的 alias 反查结果。
        self._alias_to_uuid[(workspace, alias)] = uuid_str

    def get_atom_by_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> Optional[MemoryAtom]:
        """通过指定 Workspace 分区内的别名获取缓存原子，未命中返回 None。"""
        workspace = self._require_workspace_identity(workspace_identity)
        uuid_str = self._alias_to_uuid.get((workspace, alias))
        if uuid_str is None:
            self._alias_misses += 1
            return None
        atom = self._uuid_to_atom.get(uuid_str)
        if atom is None:  # pragma: no cover - 索引不变量保护
            self._alias_misses += 1
            return None
        self._alias_hits += 1
        return atom

    def get_atom_by_uuid(self, uuid: str) -> Optional[MemoryAtom]:
        """通过 UUID 获取缓存原子；授权由调用方在资源 owner 边界重验。"""
        return self._uuid_to_atom.get(uuid)

    def has_alias(self, alias: str, *, workspace_identity: WorkspaceIdentity) -> bool:
        """检查别名是否已在指定 Workspace 分区内缓存。"""
        workspace = self._require_workspace_identity(workspace_identity)
        return (workspace, alias) in self._alias_to_uuid

    def invalidate_alias(
        self,
        alias: str,
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> None:
        """使指定 Workspace 分区内的别名缓存失效。"""
        workspace = self._require_workspace_identity(workspace_identity)
        uuid_str = self._alias_to_uuid.pop((workspace, alias), None)
        if uuid_str is None:
            return
        # 仍有其他 Workspace 分区的别名指向同一 UUID 时保留原子条目，避免
        # 留下 "has_alias 为真但反查为 None" 的悬空索引。会话级缓存规模下
        # 采用线性扫描换取结构简单；若未来规模上升，可引入
        # uuid -> set[(workspace, alias)] 反向索引恢复 O(1)。
        if not any(key_uuid == uuid_str for key_uuid in self._alias_to_uuid.values()):
            self._uuid_to_atom.pop(uuid_str, None)

    def clear(self) -> None:
        """清空会话内全部原子缓存。"""
        self._uuid_to_atom.clear()
        self._alias_to_uuid.clear()

    @property
    def size(self) -> int:
        """返回当前缓存原子数量。"""
        return len(self._uuid_to_atom)

    @property
    def alias_hits(self) -> int:
        """返回 alias 读取路径的累计命中次数。"""
        return self._alias_hits

    @property
    def alias_misses(self) -> int:
        """返回 alias 读取路径的累计未命中次数。"""
        return self._alias_misses
