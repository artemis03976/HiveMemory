"""
HiveMemory - 记忆归档器 (冷存储管理)

处理记忆在热存储和冷存储之间的迁移。

归档流程:
1. 从 Qdrant 获取记忆
2. 序列化为 JSON
3. 可选 GZIP 压缩
4. 保存到文件系统
5. 从 Qdrant 删除

唤醒流程:
1. 查询索引获取文件路径
2. 加载并反序列化
3. 写回 Qdrant
4. 删除归档文件

作者: HiveMemory Team
版本: 0.1.0
"""

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

from hivememory.core.models import MemoryAtom
from hivememory.lifecycle.interfaces import MemoryArchiver
from hivememory.lifecycle.types import ArchiveRecord, ArchiveStatus

logger = logging.getLogger(__name__)


class FileBasedMemoryArchiver(MemoryArchiver):
    """
    基于文件系统的冷存储归档器

    将归档的记忆保存为压缩的 JSON 文件，维护索引以快速查找。

    目录结构:
        data/archived/
        ├── archive_index.json      # 归档索引
        └── 2025-01/                # 按月份组织
            ├── {uuid1}.json.gz
            └── {uuid2}.json.gz

    Examples:
        >>> archiver = FileBasedMemoryArchiver(storage, archive_dir="data/archived")
        >>> archiver.archive(memory_id)
        >>> memory = archiver.resurrect(memory_id)
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        archive_dir: str = "data/archived",
        compress: bool = True,
    ):
        """
        初始化归档器

        Args:
            storage: 向量存储实例 (QdrantMemoryStore)
            archive_dir: 归档目录路径
            compress: 是否使用 GZIP 压缩
        """
        self.storage = storage
        self.archive_dir = Path(archive_dir)
        self.compress = compress

        # 创建归档目录
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        # 索引文件路径
        self.index_path = self.archive_dir / "archive_index.json"

        # 加载索引
        self._index: Dict[str, ArchiveRecord] = self._load_index()

        logger.info(
            f"FileBasedMemoryArchiver initialized: "
            f"dir={self.archive_dir}, compress={self.compress}, "
            f"indexed={len(self._index)} memories"
        )

    def archive(self, memory_id: UUID) -> None:
        """
        归档记忆到冷存储

        Args:
            memory_id: 记忆ID

        Raises:
            ValueError: 记忆不存在或已归档
        """
        # 检查是否已归档
        if str(memory_id) in self._index:
            logger.warning(f"Memory {memory_id} already archived")
            return

        # 从热存储获取记忆
        memory = self.storage.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found in hot storage")

        # 记录归档前的生命力
        original_vitality = memory.meta.vitality_score

        # 序列化为 JSON
        data = memory.model_dump(mode='json')

        # 生成存储路径
        file_path = self._get_archive_path(memory_id)

        # 写入文件
        if self.compress:
            file_path = file_path.with_suffix(".json.gz")
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # 获取文件大小
        file_size = file_path.stat().st_size if file_path.exists() else None

        # 创建归档记录
        record = ArchiveRecord(
            memory_id=memory_id,
            original_vitality=original_vitality,
            archived_at=datetime.now(),
            storage_path=str(file_path),
            compressed_size_bytes=file_size,
        )

        # 更新索引
        self._index[str(memory_id)] = record
        self._save_index()

        # 从热存储删除
        self.storage.delete_memory(memory_id)

        logger.info(
            f"Archived memory {memory_id} to {file_path.name} "
            f"({file_size} bytes, vitality={original_vitality:.2f})"
        )

    def resurrect(self, memory_id: UUID) -> MemoryAtom:
        """
        从冷存储唤醒记忆

        Args:
            memory_id: 记忆ID

        Returns:
            MemoryAtom: 唤醒的记忆

        Raises:
            ValueError: 记忆未在冷存储中
        """
        # 查询索引
        record = self._index.get(str(memory_id))
        if record is None:
            raise ValueError(f"Memory {memory_id} not found in archive")

        # 加载文件
        file_path = Path(record.storage_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Archive file not found: {file_path}")

        # 根据压缩类型选择读取方式
        if self.compress or file_path.suffix == ".gz":
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # 反序列化为 MemoryAtom
        memory = MemoryAtom(**data)

        # 写回热存储
        self.storage.upsert_memory(memory)

        # 从索引删除
        del self._index[str(memory_id)]
        self._save_index()

        # 删除归档文件
        file_path.unlink(missing_ok=True)

        logger.info(f"Resurrected memory {memory_id} to hot storage")

        return memory

    def is_archived(self, memory_id: UUID) -> bool:
        """
        检查记忆是否已归档

        Args:
            memory_id: 记忆ID

        Returns:
            bool: 是否已归档
        """
        return str(memory_id) in self._index

    def get_archive_record(self, memory_id: UUID) -> Optional[ArchiveRecord]:
        """
        获取归档记录

        Args:
            memory_id: 记忆ID

        Returns:
            Optional[ArchiveRecord]: 归档记录，不存在则返回 None
        """
        return self._index.get(str(memory_id))

    def list_archived(
        self,
        limit: int = 100,
        vitality_threshold: Optional[float] = None
    ) -> List[ArchiveRecord]:
        """
        列出已归档的记忆

        Args:
            limit: 最大返回数量
            vitality_threshold: 过滤归档时的生命力阈值

        Returns:
            List[ArchiveRecord]: 归档记录列表，最新的在前
        """
        records = list(self._index.values())

        if vitality_threshold is not None:
            records = [
                r for r in records
                if r.original_vitality <= vitality_threshold
            ]

        # 按归档时间倒序排序
        records.sort(key=lambda x: x.archived_at, reverse=True)

        return records[:limit]

    def _get_archive_path(self, memory_id: UUID) -> Path:
        """
        生成归档文件路径

        按月份组织文件，便于维护。

        Args:
            memory_id: 记忆ID

        Returns:
            Path: 文件路径
        """
        date_str = datetime.now().strftime("%Y-%m")
        date_dir = self.archive_dir / date_str
        date_dir.mkdir(exist_ok=True)
        return date_dir / f"{memory_id}.json"

    def _load_index(self) -> Dict[str, ArchiveRecord]:
        """
        加载归档索引

        Returns:
            Dict[str, ArchiveRecord]: memory_id -> ArchiveRecord
        """
        if not self.index_path.exists():
            logger.debug(f"Archive index not found, creating new one")
            return {}

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                k: ArchiveRecord(**v)
                for k, v in data.items()
            }
        except Exception as e:
            logger.error(f"Failed to load archive index: {e}")
            return {}

    def _save_index(self) -> None:
        """保存归档索引"""
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.model_dump(mode='json') for k, v in self._index.items()},
                f,
                ensure_ascii=False,
                indent=2
            )


class S3MemoryArchiver(MemoryArchiver):
    """
    基于 S3 的冷存储归档器 (预留接口)

    使用 S3 存储归档记忆，适合大规模部署。

    TODO: 实现 S3 集成
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        bucket: str,
        prefix: str = "archived/",
    ):
        """
        初始化 S3 归档器

        Args:
            storage: 向量存储实例
            bucket: S3 bucket 名称
            prefix: S3 key 前缀
        """
        raise NotImplementedError(
            "S3 archiver not yet implemented. "
            "Use FileBasedMemoryArchiver for local storage."
        )

    def archive(self, memory_id: UUID) -> None:
        """归档到 S3"""
        raise NotImplementedError

    def resurrect(self, memory_id: UUID) -> MemoryAtom:
        """从 S3 唤醒"""
        raise NotImplementedError

    def is_archived(self, memory_id: UUID) -> bool:
        """检查是否在 S3 中"""
        raise NotImplementedError

    def list_archived(self, limit: int = 100) -> List[ArchiveRecord]:
        """列出 S3 中的归档"""
        raise NotImplementedError


def create_default_archiver(
    storage,
    archive_dir: str = "data/archived",
    compress: bool = True
) -> MemoryArchiver:
    """
    创建默认归档器

    Args:
        storage: 向量存储实例
        archive_dir: 归档目录
        compress: 是否压缩

    Returns:
        MemoryArchiver: 归档器实例
    """
    return FileBasedMemoryArchiver(storage, archive_dir, compress)


__all__ = [
    "FileBasedMemoryArchiver",
    "S3MemoryArchiver",
    "create_default_archiver",
]
