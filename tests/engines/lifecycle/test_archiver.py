"""
HiveMemory - 归档器单元测试

测试内容:
- 归档记忆到冷存储
- 唤醒记忆
- 索引管理
- 检查归档状态
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.lifecycle.archiver import FileBasedMemoryArchiver


class TestFileBasedMemoryArchiver:
    """测试文件系统归档器"""

    def setup_method(self):
        """测试初始化"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

        self.mock_storage = Mock()

        self.archiver = FileBasedMemoryArchiver(
            storage=self.mock_storage,
            archive_dir=self.temp_dir,
            compress=False  # 测试时不压缩
        )

        self.test_memory = MemoryAtom(
            id=uuid4(),
            meta=MetaData(
                source_agent_id="agent1",
                user_id="user1",
                confidence_score=0.8,
                vitality_score=0.15,  # 低生命力
            ),
            index=IndexLayer(
                title="Test",
                summary="Test summary with enough length",
                tags=["test"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="Content"),
        )

    def teardown_method(self):
        """清理临时目录"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_archive_memory(self):
        """测试归档记忆"""
        self.mock_storage.get_memory.return_value = self.test_memory

        memory_id = self.test_memory.id
        self.archiver.archive(memory_id)

        # 验证从热存储删除
        self.mock_storage.delete_memory.assert_called_once_with(memory_id)

        # 验证索引中有记录
        assert self.archiver.is_archived(memory_id)

        # 验证文件存在
        record = self.archiver.get_archive_record(memory_id)
        assert Path(record.storage_path).exists()

    def test_archive_already_archived(self):
        """测试已归档记忆不重复归档"""
        self.mock_storage.get_memory.return_value = self.test_memory

        # 第一次归档
        self.archiver.archive(self.test_memory.id)

        # 第二次归档应该跳过
        self.archiver.archive(self.test_memory.id)

        # delete_memory 应该只被调用一次
        assert self.mock_storage.delete_memory.call_count == 1

    def test_resurrect_memory(self):
        """测试唤醒记忆"""
        # 先归档
        self.mock_storage.get_memory.return_value = self.test_memory
        self.archiver.archive(self.test_memory.id)

        # 重置 mock
        self.mock_storage.get_memory.reset_mock()
        self.mock_storage.upsert_memory.return_value = None

        # 唤醒
        resurrected = self.archiver.resurrect(self.test_memory.id)

        # 验证内容正确
        assert resurrected.id == self.test_memory.id
        assert resurrected.index.title == self.test_memory.index.title

        # 验证写回热存储
        self.mock_storage.upsert_memory.assert_called_once()

        # 验证从索引删除
        assert not self.archiver.is_archived(self.test_memory.id)

    def test_resurrect_nonexistent(self):
        """测试唤醒不存在的记忆抛出异常"""
        with pytest.raises(ValueError, match="not found in archive"):
            self.archiver.resurrect(uuid4())

    def test_list_archived(self):
        """测试列出已归档记忆"""
        # 归档多个记忆
        archived_ids = []
        for i in range(3):
            memory = MemoryAtom(
                id=uuid4(),
                meta=MetaData(
                    source_agent_id="agent1",
                    user_id="user1",
                    confidence_score=0.8,
                    vitality_score=0.1 + (i * 0.05),
                ),
                index=IndexLayer(
                    title=f"Test {i}",
                    summary="Summary with enough length",
                    tags=["test"],
                    memory_type=MemoryType.FACT,
                ),
                payload=PayloadLayer(content="Content"),
            )
            self.mock_storage.get_memory.return_value = memory
            self.archiver.archive(memory.id)
            archived_ids.append(memory.id)

        # 列出所有
        archived = self.archiver.list_archived()
        assert len(archived) == 3

        # 验证ID匹配
        returned_ids = [r.memory_id for r in archived]
        for aid in archived_ids:
            assert aid in returned_ids

    def test_list_archived_with_threshold(self):
        """测试按阈值过滤归档"""
        # 归档不同生命力的记忆
        for i in range(3):
            memory = MemoryAtom(
                id=uuid4(),
                meta=MetaData(
                    source_agent_id="agent1",
                    user_id="user1",
                    confidence_score=0.8,
                    vitality_score=0.1 + (i * 0.05),
                ),
                index=IndexLayer(
                    title=f"Test {i}",
                    summary="Summary for test with enough length",
                    tags=["test"],
                    memory_type=MemoryType.FACT,
                ),
                payload=PayloadLayer(content="Content"),
            )
            self.mock_storage.get_memory.return_value = memory
            self.archiver.archive(memory.id)

        # 按阈值过滤 (只返回 vitality <= 0.12)
        filtered = self.archiver.list_archived(vitality_threshold=0.12)
        assert len(filtered) == 1

    def test_archive_with_compression(self):
        """测试压缩归档"""
        # 创建支持压缩的归档器
        archiver = FileBasedMemoryArchiver(
            storage=self.mock_storage,
            archive_dir=self.temp_dir,
            compress=True
        )

        self.mock_storage.get_memory.return_value = self.test_memory

        archiver.archive(self.test_memory.id)

        # 验证文件是 .gz 格式
        record = archiver.get_archive_record(self.test_memory.id)
        assert record.storage_path.endswith(".json.gz")
        assert Path(record.storage_path).exists()

    def test_archive_index_persistence(self):
        """测试索引持久化"""
        self.mock_storage.get_memory.return_value = self.test_memory

        # 归档一个记忆
        self.archiver.archive(self.test_memory.id)

        # 创建新的归档器 (模拟重启)
        new_archiver = FileBasedMemoryArchiver(
            storage=self.mock_storage,
            archive_dir=self.temp_dir,
            compress=False
        )

        # 验证索引被加载
        assert new_archiver.is_archived(self.test_memory.id)

    def test_get_archive_record(self):
        """测试获取归档记录"""
        self.mock_storage.get_memory.return_value = self.test_memory

        # 未归档时
        assert self.archiver.get_archive_record(self.test_memory.id) is None

        # 归档后
        self.archiver.archive(self.test_memory.id)
        record = self.archiver.get_archive_record(self.test_memory.id)

        assert record is not None
        assert record.memory_id == self.test_memory.id
        assert record.original_vitality == 0.15

    def test_archive_memory_not_found(self):
        """测试归档不存在的记忆"""
        self.mock_storage.get_memory.return_value = None

        with pytest.raises(ValueError, match="not found in hot storage"):
            self.archiver.archive(uuid4())

    def test_archive_creates_date_directory(self):
        """测试归档按日期组织目录"""
        self.mock_storage.get_memory.return_value = self.test_memory

        self.archiver.archive(self.test_memory.id)

        # 验证目录结构
        record = self.archiver.get_archive_record(self.test_memory.id)
        path = Path(record.storage_path)

        # 路径应该包含日期目录 (YYYY-MM)
        # 例如: data/archived/2025-01/{uuid}.json
        assert len(path.parts) >= 3  # 至少有: archive_dir, YYYY-MM, filename
