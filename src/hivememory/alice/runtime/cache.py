"""
Kernel Runtime 缓存实现。

替代原有的双层别名映射与 _LRUCache，提供：
- 完整 MemoryAtom 对象缓存（而非仅 UUID 或代码字符串）
- 双索引：alias -> UUID 和 UUID -> MemoryAtom
- 会话级生命周期（无需 LRU 淘汰）
- UPDATE 后缓存失效支持

作者: HiveMemory Team
版本: 1.0
"""

import logging
import re
from collections import OrderedDict
from typing import Dict, List, Optional
from uuid import uuid4

from hivememory.alice.runtime.models import PendingAtom, PendingAtomStatus
from hivememory.core.models import AgentProfile, Identity, MemoryAtom, MemoryType

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 30) -> str:
    """将文本转为 alias 友好的 slug 片段。"""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len].rstrip("_")


class PendingAtomCache:
    """
    L0 运行时 pending atom 缓存。

    作为 Alice runtime 共享基础设施，主帧和子帧共享同一实例。
    子帧 WRITE 后主帧自动可见，无需 merge。
    """

    def __init__(self) -> None:
        self._atoms: Dict[str, PendingAtom] = {}

    def register_write(
        self,
        content: str,
        title: Optional[str],
        reason: Optional[str],
        identity: Identity,
        run_id: str = "",
        frame_id: str = "",
        depth: int = 0,
    ) -> PendingAtom:
        """注册 WRITE pending atom，返回带有生成 alias 的 PendingAtom。"""
        slug_source = title if title else content[:20]
        slug = _slugify(slug_source)
        if not slug:
            slug = "untitled"
        short_id = uuid4().hex[:4]
        pending_alias = f"draft_{slug}_{short_id}"

        atom = PendingAtom(
            pending_alias=pending_alias,
            status=PendingAtomStatus.PENDING,
            source_verb="WRITE",
            content=content,
            title=title,
            reason=reason,
            identity=identity,
            run_id=run_id,
            frame_id=frame_id,
            depth=depth,
        )
        self._atoms[pending_alias] = atom
        logger.debug(f"Registered pending WRITE: {pending_alias}")
        return atom

    def register_update(
        self,
        target_alias: str,
        target_uuid: str,
        instruction: str,
        content: Optional[str],
        identity: Identity,
        run_id: str = "",
        frame_id: str = "",
        depth: int = 0,
    ) -> PendingAtom:
        """注册 UPDATE pending revision，返回带有生成 alias 的 PendingAtom。"""
        short_id = uuid4().hex[:4]
        pending_alias = f"rev_{target_alias}_{short_id}"

        atom = PendingAtom(
            pending_alias=pending_alias,
            status=PendingAtomStatus.REVISION,
            source_verb="UPDATE",
            content=content or "",
            instruction=instruction,
            target_alias=target_alias,
            target_uuid=target_uuid,
            identity=identity,
            run_id=run_id,
            frame_id=frame_id,
            depth=depth,
        )
        self._atoms[pending_alias] = atom
        logger.debug(f"Registered pending UPDATE: {pending_alias}")
        return atom

    def get(self, pending_alias: str) -> Optional[PendingAtom]:
        """通过 pending alias 查询。"""
        return self._atoms.get(pending_alias)

    def has(self, alias: str) -> bool:
        """检查 alias 是否为已注册的 pending atom。"""
        return alias in self._atoms

    def all_aliases(self) -> List[str]:
        """返回所有已注册的 pending alias。"""
        return list(self._atoms.keys())

    def all_atoms(self) -> List[PendingAtom]:
        """返回所有已注册的 PendingAtom。"""
        return list(self._atoms.values())

    def clear(self) -> None:
        """清空全部 pending atom。"""
        self._atoms.clear()

    @property
    def size(self) -> int:
        """当前缓存的 pending atom 数量。"""
        return len(self._atoms)


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


class AgentProfileCache:
    """
    人偶图纸缓存 - 会话级 LRU 缓存。

    通过 alias 快速加载并缓存人偶图纸。
    缓存 Miss 时通过 storage.get_memory_by_alias 精确查找。
    """

    def __init__(self, max_size: int = 32):
        """初始化图纸缓存，默认最大 32 条。"""
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Optional[MemoryAtom], AgentProfile]] = OrderedDict()

    def get(self, alias: str) -> Optional[AgentProfile]:
        """从缓存获取配置（不触发 storage 查询）。"""
        entry = self._cache.get(alias)
        if entry is not None:
            self._cache.move_to_end(alias)
            return entry[1]
        return None

    def get_atom(self, alias: str) -> Optional[MemoryAtom]:
        """获取缓存中的完整 MemoryAtom（含 payload）。"""
        entry = self._cache.get(alias)
        if entry is not None:
            self._cache.move_to_end(alias)
            return entry[0]
        return None

    def load(self, alias: str, storage) -> Optional[AgentProfile]:
        """加载人偶图纸：缓存优先，未命中时回源 storage。"""
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

        config = AgentProfile.from_atom(atom)
        if config is None:
            return None

        self.store(alias, atom, config)
        logger.info(f"Agent profile '{alias}' loaded and cached.")
        return config

    def parse_config(self, atom: MemoryAtom) -> Optional[AgentProfile]:
        """[已废弃] 请改用 AgentProfile.from_atom"""
        return AgentProfile.from_atom(atom)

    def invalidate(self, alias: str) -> None:
        """驱逐指定别名缓存。"""
        self._cache.pop(alias, None)

    def clear(self) -> None:
        """清空全部图纸缓存。"""
        self._cache.clear()

    def store(self, alias: str, atom: Optional[MemoryAtom], config: AgentProfile) -> None:
        """写入缓存并维护 LRU 淘汰策略。"""
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
        """返回当前缓存条目数。"""
        return len(self._cache)
