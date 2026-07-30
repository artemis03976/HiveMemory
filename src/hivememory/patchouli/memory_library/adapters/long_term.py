"""
FileBasedStorageAdapter — LongTermStoragePort 的文件系统实现

将 FileBasedArchiver 的冷存储读写逻辑映射到 LongTermStoragePort 接口。
跨层操作（从 Qdrant 获取 / 写回 Qdrant）已上移至 MemoryLibrary，此适配器
仅负责文件系统的读写和索引维护。

实现阶段: Phase 1
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.models import StorageHealthComponent
from hivememory.patchouli.memory_library.ports import LongTermStoragePort

logger = logging.getLogger(__name__)


class FileBasedStorageAdapter(LongTermStoragePort):
    """
    文件系统冷存储适配器。

    只负责序列化/反序列化与文件 I/O，不感知中期存储（Qdrant）。
    索引结构与 FileBasedArchiver 保持兼容以便平滑迁移。
    """

    def __init__(self, archive_dir: str, compress: bool = True) -> None:
        self._archive_dir = Path(archive_dir)
        self._compress = compress
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._archive_dir / "archive_index.json"
        self._index: Dict[str, ArchiveRecord] = self._load_index()
        logger.info(
            f"FileBasedStorageAdapter 初始化: dir={self._archive_dir}, "
            f"compress={compress}, indexed={len(self._index)}"
        )

    # ── LongTermStoragePort ──

    async def persist(self, memory: MemoryAtom) -> None:
        """将 MemoryAtom 序列化写入文件，更新索引。不负责从中期删除。"""
        mid = str(memory.id)
        if mid in self._index:
            logger.warning(f"Memory {memory.id} already in cold storage, overwriting")

        data = memory.model_dump(mode="json")
        file_path = self._get_file_path(memory.id)

        if self._compress:
            file_path = file_path.with_suffix(".json.gz")
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        self._index[mid] = ArchiveRecord(
            memory_id=memory.id,
            original_vitality=memory.meta.vitality_score,
            archived_at=datetime.now(),
            storage_path=str(file_path),
            compressed_size_bytes=file_path.stat().st_size if file_path.exists() else None,
        )
        self._save_index()
        logger.info(f"持久化记忆到冷存储: {memory.id}")

    async def load(self, memory_id: UUID) -> MemoryAtom:
        """从文件加载 MemoryAtom，不负责写回中期存储。"""
        record = self._index.get(str(memory_id))
        if record is None:
            raise ValueError(f"Memory {memory_id} not found in cold storage")

        file_path = Path(record.storage_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archive file not found: {file_path}")

        if self._compress or file_path.suffix == ".gz":
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return MemoryAtom(**data)

    async def remove(self, memory_id: UUID) -> None:
        """从索引和文件系统删除归档记录。"""
        record = self._index.pop(str(memory_id), None)
        if record is None:
            return
        self._save_index()
        Path(record.storage_path).unlink(missing_ok=True)
        logger.info(f"从冷存储删除记忆: {memory_id}")

    async def is_archived(self, memory_id: UUID) -> bool:
        return str(memory_id) in self._index

    async def query(
        self,
        limit: int = 100,
        vitality_threshold: Optional[float] = None,
    ) -> List[ArchiveRecord]:
        records = list(self._index.values())
        if vitality_threshold is not None:
            records = [r for r in records if r.original_vitality <= vitality_threshold]
        records.sort(key=lambda r: r.archived_at, reverse=True)
        return records[:limit]

    async def check_health(self) -> StorageHealthComponent:
        try:
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            probe = self._archive_dir / ".healthcheck"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return StorageHealthComponent(name="long_term", healthy=True)
        except Exception as exc:
            return StorageHealthComponent(
                name="long_term",
                healthy=False,
                detail=str(exc),
            )

    # ── 内部辅助 ──

    def _get_file_path(self, memory_id: UUID) -> Path:
        date_dir = self._archive_dir / datetime.now().strftime("%Y-%m")
        date_dir.mkdir(exist_ok=True)
        return date_dir / f"{memory_id}.json"

    def _load_index(self) -> Dict[str, ArchiveRecord]:
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: ArchiveRecord(**v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载归档索引失败: {e}")
            return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.model_dump(mode="json") for k, v in self._index.items()},
                f,
                ensure_ascii=False,
                indent=2,
            )


__all__ = ["FileBasedStorageAdapter"]
