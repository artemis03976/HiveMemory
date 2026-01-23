"""
HiveMemory - 垃圾回收器

定期扫描低生命力记忆并触发归档。

"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from uuid import UUID
from datetime import datetime
import logging

from hivememory.engines.lifecycle.interfaces import (
    GarbageCollector,
    LifecycleManager,
    VitalityCalculator,
    MemoryArchiver
)
from hivememory.engines.lifecycle.models import ArchiveStatus
from hivememory.infrastructure.storage import QdrantMemoryStore

if TYPE_CHECKING:
    from hivememory.patchouli.config import GarbageCollectorConfig

logger = logging.getLogger(__name__)


class PeriodicGarbageCollector(GarbageCollector):
    """
    周期性垃圾回收器

    扫描低生命力记忆并批量归档。

    工作流程:
        1. 扫描所有记忆
        2. 计算生命力分数
        3. 筛选低于阈值的记忆
        4. 批量归档

    Examples:
        >>> gc = PeriodicGarbageCollector(storage, archiver, vitality_calculator)
        >>> archived_count = gc.collect(force=True)
        >>> print(f"Archived {archived_count} memories")
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        archiver: MemoryArchiver,
        vitality_calculator: VitalityCalculator,
        low_watermark: float = 20.0,
        batch_size: int = 10,
    ):
        """
        初始化垃圾回收器

        Args:
            storage: 向量存储实例 (QdrantMemoryStore)
            archiver: 归档器实例
            vitality_calculator: 生命力计算器实例
            low_watermark: 低水位阈值 (0-100)
            batch_size: 每次最多归档数量
        """
        self.storage = storage
        self.archiver = archiver
        self.vitality_calculator = vitality_calculator
        self.low_watermark = low_watermark
        self.batch_size = batch_size

        # 统计信息
        self._stats: Dict[str, Any] = {
            "last_run": None,
            "total_scanned": 0,
            "total_archived": 0,
            "total_skipped": 0,
            "runs_count": 0,
        }

        logger.info(
            f"PeriodicGarbageCollector initialized: "
            f"threshold={low_watermark}, batch_size={batch_size}"
        )

    def scan_candidates(self, vitality_threshold: Optional[float] = None) -> List[UUID]:
        """
        扫描低于生命力阈值的记忆

        Args:
            vitality_threshold: 自定义阈值 (默认使用 low_watermark)

        Returns:
            List[UUID]: 候选记忆ID列表
        """
        threshold = vitality_threshold if vitality_threshold is not None else self.low_watermark

        logger.info(f"Scanning for memories with vitality <= {threshold}...")

        # 获取所有记忆
        all_memories = self.storage.get_all_memories()

        candidates = []
        for memory in all_memories:
            # 计算当前生命力
            vitality = self.vitality_calculator.calculate(memory)

            # 归一化阈值比较 (配置是 0-100，计算结果也是 0-100)
            if vitality <= threshold:
                candidates.append((memory.id, vitality))

        # 按生命力排序 (最低的优先)
        candidates.sort(key=lambda x: x[1])

        logger.info(f"Found {len(candidates)} candidates for archival")

        return [mid for mid, _ in candidates]

    def collect(
        self,
        force: bool = False,
        batch_size: Optional[int] = None,
        vitality_threshold: Optional[float] = None
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

        # 扫描候选
        candidate_ids = self.scan_candidates(vitality_threshold=vitality_threshold)

        if not candidate_ids:
            logger.info("No candidates found for archival")
            self._update_stats(0, 0)
            return 0

        # 限制批量大小
        actual_batch_size = batch_size or self.batch_size
        candidate_ids = candidate_ids[:actual_batch_size]

        archived_count = 0
        skipped_count = 0

        for memory_id in candidate_ids:
            try:
                # 检查是否已归档
                if hasattr(self.archiver, "is_archived"):
                    if self.archiver.is_archived(memory_id):
                        logger.debug(f"Memory {memory_id} already archived")
                        skipped_count += 1
                        continue

                # 执行归档
                self.archiver.archive(memory_id)
                archived_count += 1

            except Exception as e:
                logger.error(f"Failed to archive {memory_id}: {e}")
                skipped_count += 1

        # 更新统计
        self._update_stats(len(candidate_ids), archived_count, skipped_count)

        logger.info(
            f"Garbage collection complete: "
            f"{archived_count} archived, {skipped_count} skipped"
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
        """重置统计信息"""
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
        skipped: int = 0
    ) -> None:
        """
        更新统计信息

        Args:
            scanned: 扫描的记忆数量
            archived: 归档的记忆数量
            skipped: 跳过的记忆数量
        """
        self._stats["last_run"] = datetime.now().isoformat()
        self._stats["total_scanned"] += scanned
        self._stats["total_archived"] += archived
        self._stats["total_skipped"] += skipped
        self._stats["runs_count"] += 1


class ScheduledGarbageCollector(PeriodicGarbageCollector):
    """
    支持定时触发的垃圾回收器

    使用 APScheduler 实现定时任务。

    注意: 需要安装 apscheduler 包
    """

    def __init__(
        self,
        storage,
        archiver: MemoryArchiver,
        vitality_calculator: VitalityCalculator,
        low_watermark: float = 20.0,
        batch_size: int = 10,
        enable_schedule: bool = False,
        interval_hours: int = 24,
    ):
        """
        初始化定时垃圾回收器

        Args:
            storage: 向量存储实例
            archiver: 归档器实例
            vitality_calculator: 生命力计算器实例
            low_watermark: 低水位阈值
            batch_size: 批量大小
            enable_schedule: 是否启用定时执行
            interval_hours: 执行间隔 (小时)
        """
        super().__init__(
            storage=storage,
            archiver=archiver,
            vitality_calculator=vitality_calculator,
            low_watermark=low_watermark,
            batch_size=batch_size,
        )

        self.enable_schedule = enable_schedule
        self.interval_hours = interval_hours
        self._scheduler = None

        if enable_schedule:
            self._setup_schedule()

    def _setup_schedule(self) -> None:
        """设置定时任务"""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()

            # 添加定时任务
            self._scheduler.add_job(
                self.collect,
                "interval",
                hours=self.interval_hours,
                id="garbage_collection",
                replace_existing=True,
            )

            self._scheduler.start()
            logger.info(
                f"Scheduled garbage collection: interval={self.interval_hours}h"
            )
        except ImportError:
            logger.warning(
                "apscheduler not installed, scheduled GC disabled. "
                "Install with: pip install apscheduler"
            )
            self.enable_schedule = False

    def shutdown(self) -> None:
        """关闭调度器"""
        if self._scheduler:
            self._scheduler.shutdown()
            logger.info("Scheduler shutdown")


def create_default_garbage_collector(
    storage,
    archiver: MemoryArchiver,
    vitality_calculator: VitalityCalculator,
    config: Optional["GarbageCollectorConfig"] = None,
) -> GarbageCollector:
    """
    创建默认垃圾回收器

    Args:
        storage: 向量存储实例
        archiver: 归档器实例
        vitality_calculator: 生命力计算器实例
        config: 垃圾回收器配置 (可选)

    Returns:
        GarbageCollector: 垃圾回收器实例
    """
    if config is None:
        from hivememory.patchouli.config import GarbageCollectorConfig
        config = GarbageCollectorConfig()

    return PeriodicGarbageCollector(
        storage=storage,
        archiver=archiver,
        vitality_calculator=vitality_calculator,
        low_watermark=config.low_watermark,
        batch_size=config.batch_size,
    )


__all__ = [
    "PeriodicGarbageCollector",
    "ScheduledGarbageCollector",
    "create_default_garbage_collector",
]
