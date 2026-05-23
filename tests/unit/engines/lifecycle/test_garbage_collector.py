"""Unit tests for the lifecycle garbage collector."""

from unittest.mock import Mock
from uuid import uuid4

from hivememory.core.models import (
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.engines.lifecycle.garbage_collector import PeriodicGarbageCollector
from hivememory.system.config import GarbageCollectorConfig


def _make_memory(title: str, vitality_score: float | None) -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(
            source_agent_id="a",
            user_id="u",
            vitality_score=vitality_score,
            confidence_score=0.8,
        ),
        index=IndexLayer(
            title=title,
            summary=f"summary for {title}",
            tags=[],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="content"),
    )


class TestPeriodicGarbageCollector:
    def setup_method(self):
        self.mock_archiver = Mock()
        self.config = GarbageCollectorConfig(
            low_watermark=20.0,
            batch_size=10,
        )
        self.gc = PeriodicGarbageCollector(
            archiver=self.mock_archiver,
            config=self.config,
        )
        self.low_vitality_memory = _make_memory("Low", 10.0)
        self.high_vitality_memory = _make_memory("High", 90.0)

    def test_scan_candidates(self):
        candidates = self.gc.scan_candidates(
            [self.low_vitality_memory, self.high_vitality_memory],
            vitality_threshold=20.0,
        )

        assert candidates == [self.low_vitality_memory.id]

    def test_scan_candidates_uses_default_threshold(self):
        candidates = self.gc.scan_candidates([self.low_vitality_memory])

        assert candidates == [self.low_vitality_memory.id]

    def test_scan_candidates_sorts_lowest_first(self):
        lower = _make_memory("Lower", 5.0)
        higher = _make_memory("Higher", 15.0)

        candidates = self.gc.scan_candidates([higher, lower])

        assert candidates == [lower.id, higher.id]

    def test_scan_candidates_skips_missing_vitality(self):
        missing = Mock()
        missing.id = uuid4()
        missing.meta.vitality_score = None

        candidates = self.gc.scan_candidates([missing])

        assert candidates == []

    def test_collect_archives_candidates(self):
        self.mock_archiver.is_archived.return_value = False

        archived = self.gc.collect([self.low_vitality_memory])

        assert archived == 1
        self.mock_archiver.archive.assert_called_once_with(
            self.low_vitality_memory.id
        )

    def test_collect_skips_already_archived(self):
        self.mock_archiver.is_archived.return_value = True

        archived = self.gc.collect([self.low_vitality_memory])

        assert archived == 0
        self.mock_archiver.archive.assert_not_called()

    def test_collect_respects_batch_size(self):
        memories = [_make_memory(f"M{i}", 15.0) for i in range(20)]
        self.mock_archiver.is_archived.return_value = False

        archived = self.gc.collect(memories, batch_size=10)

        assert archived == 10
        assert self.mock_archiver.archive.call_count == 10

    def test_collect_no_candidates(self):
        archived = self.gc.collect([])

        assert archived == 0

    def test_get_stats(self):
        stats = self.gc.get_stats()

        assert "last_run" in stats
        assert "total_scanned" in stats
        assert "total_archived" in stats
        assert "total_skipped" in stats
        assert "runs_count" in stats

    def test_reset_stats(self):
        self.mock_archiver.is_archived.return_value = False
        self.gc.collect([self.low_vitality_memory])

        self.gc.reset_stats()

        stats = self.gc.get_stats()
        assert stats["total_scanned"] == 0
        assert stats["total_archived"] == 0
        assert stats["runs_count"] == 0

    def test_collect_updates_stats(self):
        self.mock_archiver.is_archived.return_value = False

        self.gc.collect([self.low_vitality_memory, self.high_vitality_memory])

        stats = self.gc.get_stats()
        assert stats["last_run"] is not None
        assert stats["total_scanned"] == 2
        assert stats["total_archived"] == 1
        assert stats["runs_count"] == 1

    def test_collect_with_custom_threshold(self):
        medium_vitality_memory = _make_memory("Medium", 30.0)
        self.mock_archiver.is_archived.return_value = False

        archived = self.gc.collect(
            [medium_vitality_memory],
            batch_size=10,
            vitality_threshold=50.0,
        )

        assert archived == 1
