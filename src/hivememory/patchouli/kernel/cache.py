"""
KoakumaAtomCache - 统一的记忆原子缓存与别名解析

替代原有的双层别名映射与 _LRUCache，提供：
- 完整 MemoryAtom 对象缓存（而非仅 UUID 或代码字符串）
- 双索引：alias -> UUID 和 UUID -> MemoryAtom
- 会话级生命周期（无需 LRU 淘汰）
- UPDATE 后缓存失效支持

作者: HiveMemory Team
版本: 1.0
"""

from typing import Dict, List, Optional
from hivememory.core.models import MemoryAtom


class KoakumaAtomCache:
    """
    统一的记忆原子缓存，带别名解析功能

    会话级缓存，存储完整的 MemoryAtom 对象。
    消除 SEARCH/READ/RUN 流程中的冗余数据库查询。
    """

    def __init__(self):
        # 核心缓存：UUID -> MemoryAtom
        self._uuid_to_atom: Dict[str, MemoryAtom] = {}
        # 别名映射：alias -> UUID
        self._alias_to_uuid: Dict[str, str] = {}

    def ingest_atoms(self, atoms: List[MemoryAtom]) -> None:
        """
        批量缓存原子并注册别名

        用于 SEARCH 和预检索结果的批量注册。

        Args:
            atoms: MemoryAtom 对象列表
        """
        for atom in atoms:
            uuid_str = str(atom.id)
            alias = atom.get_alias()
            self._uuid_to_atom[uuid_str] = atom
            self._alias_to_uuid[alias] = uuid_str

    def ingest_atom(self, atom: MemoryAtom) -> None:
        """
        缓存单个原子并注册别名

        用于 L2 冷检索命中后的缓存。

        Args:
            atom: MemoryAtom 对象
        """
        uuid_str = str(atom.id)
        alias = atom.get_alias()
        self._uuid_to_atom[uuid_str] = atom
        self._alias_to_uuid[alias] = uuid_str

    def get_atom_by_alias(self, alias: str) -> Optional[MemoryAtom]:
        """
        通过别名获取缓存的原子

        Args:
            alias: 语义化别名

        Returns:
            MemoryAtom 对象，未命中返回 None
        """
        uuid_str = self._alias_to_uuid.get(alias)
        if uuid_str is None:
            return None
        return self._uuid_to_atom.get(uuid_str)

    def get_atom_by_uuid(self, uuid: str) -> Optional[MemoryAtom]:
        """
        通过 UUID 获取缓存的原子

        Args:
            uuid: UUID 字符串

        Returns:
            MemoryAtom 对象，未命中返回 None
        """
        return self._uuid_to_atom.get(uuid)

    def has_alias(self, alias: str) -> bool:
        """
        检查别名是否已缓存

        Args:
            alias: 语义化别名

        Returns:
            True 如果别名已缓存
        """
        return alias in self._alias_to_uuid

    def invalidate_alias(self, alias: str) -> None:
        """
        使别名对应的缓存失效

        用于 UPDATE 指令后防止脏读。

        Args:
            alias: 要失效的别名
        """
        uuid_str = self._alias_to_uuid.pop(alias, None)
        if uuid_str:
            self._uuid_to_atom.pop(uuid_str, None)

    def clear(self) -> None:
        """清空所有缓存（新会话时调用）"""
        self._uuid_to_atom.clear()
        self._alias_to_uuid.clear()

    @property
    def size(self) -> int:
        """缓存的原子数量"""
        return len(self._uuid_to_atom)
