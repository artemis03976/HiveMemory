"""
Agent Runtime alias 记忆原子缓存实现。

提供：
- 完整 MemoryAtom 对象缓存
- 双索引：alias -> UUID 和 UUID -> MemoryAtom
- 会话级生命周期（无需 LRU 淘汰）
- UPDATE 后缓存失效支持

作者: HiveMemory Team
版本: 1.0
"""

import logging
from typing import Dict, List, Optional

from hivememory.core.models import MemoryAtom

logger = logging.getLogger(__name__)


class KoakumaAtomCache:
    """
    统一的记忆原子缓存，带别名解析功能

    会话级缓存，存储完整的 MemoryAtom 对象。
    消除 SEARCH/READ/RUN 流程中的冗余数据库查询。
    """

    def __init__(self):
        """初始化双索引缓存结构。"""
        # 核心缓存：UUID -> MemoryAtom
        self._uuid_to_atom: Dict[str, MemoryAtom] = {}
        # 别名映射：alias -> UUID
        self._alias_to_uuid: Dict[str, str] = {}

    def ingest_atoms(self, atoms: List[MemoryAtom]) -> None:
        """
        批量缓存原子并注册别名。

        用于 SEARCH 和预检索结果的批量注册。
        """
        for atom in atoms:
            uuid_str = str(atom.id)
            alias = atom.get_alias()
            self._uuid_to_atom[uuid_str] = atom
            self._alias_to_uuid[alias] = uuid_str

    def ingest_atom(self, atom: MemoryAtom) -> None:
        """缓存单个原子并注册别名。"""
        uuid_str = str(atom.id)
        alias = atom.get_alias()
        self._uuid_to_atom[uuid_str] = atom
        self._alias_to_uuid[alias] = uuid_str

    def get_atom_by_alias(self, alias: str) -> Optional[MemoryAtom]:
        """通过别名获取缓存原子，未命中返回 None。"""
        uuid_str = self._alias_to_uuid.get(alias)
        if uuid_str is None:
            return None
        return self._uuid_to_atom.get(uuid_str)

    def get_atom_by_uuid(self, uuid: str) -> Optional[MemoryAtom]:
        """通过 UUID 获取缓存原子。"""
        return self._uuid_to_atom.get(uuid)

    def has_alias(self, alias: str) -> bool:
        """检查别名是否已缓存。"""
        return alias in self._alias_to_uuid

    def invalidate_alias(self, alias: str) -> None:
        """使指定别名及其对应 UUID 缓存失效。"""
        uuid_str = self._alias_to_uuid.pop(alias, None)
        if uuid_str:
            self._uuid_to_atom.pop(uuid_str, None)

    def clear(self) -> None:
        """清空会话内全部原子缓存。"""
        self._uuid_to_atom.clear()
        self._alias_to_uuid.clear()

    @property
    def size(self) -> int:
        """返回当前缓存原子数量。"""
        return len(self._uuid_to_atom)
