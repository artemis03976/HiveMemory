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

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

from hivememory.core.models import MemoryAtom, MemoryType, AgentProfileConfig

logger = logging.getLogger(__name__)


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


class AgentProfileCache:
    """
    人偶图纸缓存 - 会话级 LRU 缓存

    通过 alias 快速加载并缓存人偶图纸。
    缓存 Miss 时通过 storage 的 get_memory_by_alias 精确查找。

    生命周期：与 PatchouliKernel 实例绑定，会话结束时销毁。
    """

    def __init__(self, max_size: int = 32):
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[MemoryAtom, AgentProfileConfig]] = OrderedDict()

    def get(self, alias: str) -> Optional[AgentProfileConfig]:
        """从缓存获取人偶配置（不触发 storage 查询）"""
        entry = self._cache.get(alias)
        if entry is not None:
            self._cache.move_to_end(alias)
            return entry[1]
        return None

    def get_atom(self, alias: str) -> Optional[MemoryAtom]:
        """获取缓存的完整 MemoryAtom（含灵魂 payload.content）"""
        entry = self._cache.get(alias)
        if entry is not None:
            self._cache.move_to_end(alias)
            return entry[0]
        return None

    def load(self, alias: str, storage) -> Optional[AgentProfileConfig]:
        """加载人偶图纸：缓存优先 → storage 冷查询"""
        cached = self.get(alias)
        if cached is not None:
            return cached

        try:
            atom = storage.get_memory_by_alias(alias)
        except Exception as e:
            logger.warning(f"Failed to load agent profile '{alias}' from storage: {e}")
            return None

        if atom is None:
            return None

        if atom.index.memory_type != MemoryType.AGENT_PROFILE:
            logger.warning(
                f"Alias '{alias}' resolved to type '{atom.index.memory_type}', "
                f"expected AGENT_PROFILE. Ignoring."
            )
            return None

        config = self.parse_config(atom)
        if config is None:
            return None

        self._put(alias, atom, config)
        logger.info(f"Agent profile '{alias}' loaded and cached.")
        return config

    def parse_config(self, atom: MemoryAtom) -> Optional[AgentProfileConfig]:
        """从 MemoryAtom 解析 AgentProfileConfig"""
        raw = atom.payload.artifacts.agent_config
        if raw is None:
            logger.warning(f"Agent profile atom '{atom.get_alias()}' has no agent_config in artifacts.")
            return None

        try:
            return AgentProfileConfig(**raw)
        except Exception as e:
            logger.error(f"Failed to parse agent_config for '{atom.get_alias()}': {e}")
            return None

    def invalidate(self, alias: str) -> None:
        """驱逐指定别名的缓存"""
        self._cache.pop(alias, None)

    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()

    def _put(self, alias: str, atom: MemoryAtom, config: AgentProfileConfig) -> None:
        """写入缓存，超出容量时淘汰最久未使用的条目"""
        if alias in self._cache:
            self._cache.move_to_end(alias)
            self._cache[alias] = (atom, config)
        else:
            if len(self._cache) >= self._max_size:
                evicted_alias, _ = self._cache.popitem(last=False)
                logger.debug(f"Agent profile cache evicted: '{evicted_alias}'")
            self._cache[alias] = (atom, config)

    @property
    def size(self) -> int:
        """缓存条目数"""
        return len(self._cache)
