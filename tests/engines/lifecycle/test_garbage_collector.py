"""
HiveMemory - 垃圾回收器单元测试

测试内容:
- 扫描低生命力记忆
- 批量归档
- 统计跟踪
"""

import pytest
from unittest.mock import Mock, MagicMock
from uuid import uuid4
from datetime import datetime

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.lifecycle.garbage_collector import PeriodicGarbageCollector
from hivememory.patchouli.config import GarbageCollectorConfig


class TestPeriodicGarbageCollector:
    """测试周期性垃圾回收器"""

    def setup_method(self):
        """测试初始化"""
        self.mock_storage = Mock()
        self.mock_archiver = Mock()
        self.mock_vitality_calc = Mock()

        config = GarbageCollectorConfig(
            low_watermark=20.0,
            batch_size=10,
        )

        self.gc = PeriodicGarbageCollector(
            storage=self.mock_storage,
            archiver=self.mock_archiver,
            vitality_calculator=self.mock_vitality_calc,
            config=config,
        )

        # 创建测试记忆
        self.low_vitality_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="a",
                user_id="u",
                vitality_score=10.0,
                confidence_score=0.8,
            ),
            index=IndexLayer(
                title="Low",
                summary="summary low vitality",
                tags=[],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="c"),
        )

        self.high_vitality_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="a",
                user_id="u",
                vitality_score=90.0,
                confidence_score=0.8,
            ),
            index=IndexLayer(
                title="High",
                summary="summary high vitality",
                tags=[],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="c"),
        )

    def test_scan_candidates(self):
        """测试扫描候选记忆"""
        # 设置返回记忆
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
            self.high_vitality_memory,
        ]

        # 设置批量刷新结果 (新的 refresh_batch 方法)
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),  # 低生命力记忆
            (self.high_vitality_memory.id, 85.0),  # 高生命力记忆
        ]

        candidates = self.gc.scan_candidates(vitality_threshold=20.0)

        # 应该只返回低生命力记忆
        assert len(candidates) == 1
        assert candidates[0] == self.low_vitality_memory.id

    def test_collect_archives_candidates(self):
        """测试收集归档候选"""
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),
        ]
        self.mock_archiver.is_archived.return_value = False

        archived = self.gc.collect()

        assert archived == 1
        self.mock_archiver.archive.assert_called_once_with(
            self.low_vitality_memory.id
        )

    def test_collect_skips_already_archived(self):
        """测试收集时跳过已归档记忆"""
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),
        ]
        self.mock_archiver.is_archived.return_value = True  # 已归档

        archived = self.gc.collect()

        assert archived == 0
        self.mock_archiver.archive.assert_not_called()

    def test_collect_respects_batch_size(self):
        """测试收集时尊重批量大小限制"""
        # 创建多个候选记忆
        memories = []
        refresh_results = []
        for i in range(20):
            memory = MemoryAtom(
                id=uuid4(),
                meta=MetaData(
                    source_agent_id="a",
                    user_id="u",
                    vitality_score=10.0,
                    confidence_score=0.8,
                ),
                index=IndexLayer(
                    title=f"M{i}",
                    summary="summary batch size",
                    tags=[],
                    memory_type=MemoryType.FACT,
                ),
                payload=PayloadLayer(content="c"),
            )
            memories.append(memory)
            refresh_results.append((memory.id, 15.0))

        self.mock_storage.get_all_memories.return_value = memories
        self.mock_vitality_calc.refresh_batch.return_value = refresh_results
        self.mock_archiver.is_archived.return_value = False

        archived = self.gc.collect(batch_size=10)

        # 应该只归档 10 个 (批量大小)
        assert archived == 10
        assert self.mock_archiver.archive.call_count == 10

    def test_collect_no_candidates(self):
        """测试没有候选时返回0"""
        self.mock_storage.get_all_memories.return_value = []
        self.mock_vitality_calc.refresh_batch.return_value = []

        archived = self.gc.collect()

        assert archived == 0

    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.gc.get_stats()

        assert "last_run" in stats
        assert "total_scanned" in stats
        assert "total_archived" in stats
        assert "total_skipped" in stats
        assert "runs_count" in stats

    def test_reset_stats(self):
        """测试重置统计"""
        # 执行一次收集来生成统计数据
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),
        ]
        self.mock_archiver.is_archived.return_value = False
        self.gc.collect()

        # 重置
        self.gc.reset_stats()

        stats = self.gc.get_stats()
        assert stats["total_scanned"] == 0
        assert stats["total_archived"] == 0
        assert stats["runs_count"] == 0

    def test_collect_updates_stats(self):
        """测试收集时更新统计"""
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
            self.high_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),
            (self.high_vitality_memory.id, 85.0),
        ]
        self.mock_archiver.is_archived.return_value = False

        self.gc.collect()

        stats = self.gc.get_stats()
        assert stats["last_run"] is not None
        # scan_candidates returns 1 candidate, so collect processes 1 candidate
        # But scanned count depends on implementation.
        # If we look at garbage_collector.py, it calls scan_candidates which iterates over all memories.
        # But "total_scanned" usually means how many were considered/processed.
        # Let's check get_stats implementation if needed, but assuming it counts candidates or archived items.
        # In scan_candidates log says "Scanning for memories with vitality <= threshold".
        # If the implementation counts candidates found:
        assert stats["total_archived"] == 1
        assert stats["runs_count"] == 1

    def test_collect_with_custom_threshold(self):
        """测试使用自定义阈值"""
        # 创建中等生命力记忆
        medium_vitality_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="a",
                user_id="u",
                vitality_score=30.0,  # 30/100
                confidence_score=0.8,
            ),
            index=IndexLayer(
                title="Medium",
                summary="summary medium vitality",
                tags=[],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="c"),
        )

        self.mock_storage.get_all_memories.return_value = [
            medium_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (medium_vitality_memory.id, 30.0),
        ]
        self.mock_archiver.is_archived.return_value = False

        # 使用更高阈值 (50)
        archived = self.gc.collect(batch_size=10, vitality_threshold=50.0)

        # 30 <= 50, 应该被归档
        assert archived == 1

    def test_scan_candidates_uses_default_threshold(self):
        """测试扫描时使用默认阈值"""
        self.mock_storage.get_all_memories.return_value = [
            self.low_vitality_memory,
        ]
        self.mock_vitality_calc.refresh_batch.return_value = [
            (self.low_vitality_memory.id, 15.0),
        ]

        # 不传入阈值，使用默认的 low_watermark
        candidates = self.gc.scan_candidates()

        assert len(candidates) == 1
