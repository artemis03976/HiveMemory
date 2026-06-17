"""
HiveMemory - 垃圾回收器

扫描低生命力记忆并触发归档。

"""

from datetime import datetime
import logging
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.interfaces import (
    BaseGarbageCollector,
    BaseMemoryArchiver,
)
from hivememory.system.config import GarbageCollectorConfig

logger = logging.getLogger(__name__)


class PeriodicGarbageCollector(BaseGarbageCollector):
    """
    垃圾回收器

    扫描调用方传入的低生命力记忆并批量归档。
    生命力刷新由 MemoryLifecycleEngine 或调用方在进入 GC 前完成。

    工作流程:
        1. 接收已刷新生命力的记忆集合
        2. 筛选低于阈值的记忆
        3. 批量归档

    Examples:
        >>> gc = PeriodicGarbageCollector(archiver, config)
        >>> archived_count = await gc.collect(refreshed_memories, force=True)
        >>> print(f"Archived {archived_count} memories")
    """

    def __init__(
        self,
        archiver: BaseMemoryArchiver,
        config: GarbageCollectorConfig,
    ):
        self.archiver = archiver
        self.config = config
        self._stats: Dict[str, Any] = {
            "last_run": None,
            "total_scanned": 0,
            "total_archived": 0,
            "total_skipped": 0,
            "runs_count": 0,
        }

        logger.info(
            f"PeriodicGarbageCollector initialized: threshold={config.low_watermark}, batch_size={config.batch_size}",
        )

    def scan_candidates(
        self,
        memories: Iterable[MemoryAtom],
        vitality_threshold: Optional[float] = None,
    ) -> List[UUID]:
        """
        扫描低生命力记忆

        Args:
            memories: 记忆迭代器
            vitality_threshold: 覆盖默认生命力阈值

        Returns:
            List[UUID]: 低生命力记忆ID列表
        """
        threshold = (
            vitality_threshold
            if vitality_threshold is not None
            else self.config.low_watermark
        )
        logger.info(f"Scanning for memories with vitality <= {threshold}...")

        candidates = [
            (memory.id, memory.meta.vitality_score)
            for memory in memories
            if memory.meta.vitality_score is not None
            and memory.meta.vitality_score <= threshold
        ]
        candidates.sort(key=lambda item: item[1])

        logger.info(f"Found {len(candidates)} candidates for archival")
        return [memory_id for memory_id, _ in candidates]

    async def collect(
        self,
        memories: Iterable[MemoryAtom],
        force: bool = False,
        batch_size: Optional[int] = None,
        vitality_threshold: Optional[float] = None,
    ) -> int:
        """
        运行垃圾回收

        Args:
            force: 强制执行，忽略调度限制
            batch_size: 覆盖默认批量大小
            vitality_threshold: 覆盖默认生命力阈值

        Returns:
            int: 归档的记忆数量
        """
        logger.info("Starting garbage collection...")
        memories = list(memories)
        candidate_ids = self.scan_candidates(
            memories,
            vitality_threshold=vitality_threshold,
        )

        if not candidate_ids:
            logger.info("No candidates found for archival")
            self._update_stats(len(memories), 0)
            return 0

        actual_batch_size = batch_size or self.config.batch_size
        candidate_ids = candidate_ids[:actual_batch_size]

        archived_count = 0
        skipped_count = 0

        for memory_id in candidate_ids:
            try:
                if hasattr(self.archiver, "is_archived"):
                    if self.archiver.is_archived(memory_id):
                        logger.debug("Memory %s already archived", memory_id)
                        skipped_count += 1
                        continue

                await self.archiver.archive(memory_id)
                archived_count += 1
                logger.info(f"Successfully archived {memory_id}")
            except Exception as exc:
                logger.error(f"Failed to archive {memory_id}: {exc}")
                skipped_count += 1

        self._update_stats(len(memories), archived_count, skipped_count)
        logger.info(
            f"Garbage collection complete: {archived_count} archived, {skipped_count} skipped",
        )
        return archived_count

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        self._stats = {
            "last_run": None,
            "total_scanned": 0,
            "total_archived": 0,
            "total_skipped": 0,
            "runs_count": 0,
        }
        logger.info("Statistics reset")

    def _update_stats(
        self,
        scanned: int,
        archived: int,
        skipped: int = 0,
    ) -> None:
        self._stats["last_run"] = datetime.now().isoformat()
        self._stats["total_scanned"] += scanned
        self._stats["total_archived"] += archived
        self._stats["total_skipped"] += skipped
        self._stats["runs_count"] += 1


def create_garbage_collector(
    archiver: BaseMemoryArchiver,
    config: GarbageCollectorConfig,
) -> BaseGarbageCollector:
    """
    创建默认垃圾回收器

    Args:
        archiver: 归档器实例
        config: 垃圾回收器配置

    Returns:
        BaseGarbageCollector: 垃圾回收器实例
    """
    return PeriodicGarbageCollector(
        archiver=archiver,
        config=config,
    )


__all__ = [
    "PeriodicGarbageCollector",
    "create_garbage_collector",
]
