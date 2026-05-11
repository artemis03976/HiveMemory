"""
LibrarianCore 单元测试

测试覆盖:
- 初始化: 有/无依赖注入时的行为
- 观察者: 添加 / 移除 / 移除不存在的
- _on_generate_memory: Mode A/B/C 分支 / 空消息 / 异常隔离
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4

from hivememory.core.models import Identity
from hivememory.engines.perception.models import FlushReason, LogicalBlock, ArchivePayload
from hivememory.engines.generation.models import GenerationRequest, WriteFocus, UpdateFocus
from hivememory.patchouli.kernel.librarian_core import LibrarianCore


def _make_identity() -> Identity:
    return Identity(user_id="u1", agent_id="a1")


def _make_logical_blocks(n=2):
    """创建测试用的 LogicalBlock 列表"""
    blocks = []
    for i in range(n):
        block = LogicalBlock(
            user_query=f"user_msg_{i}",
            assistant_final_text=f"assistant_msg_{i}",
            identity=_make_identity(),
        )
        blocks.append(block)
    return blocks


def _make_kernel_logical_blocks(n=2):
    blocks = []
    for i in range(n):
        block = LogicalBlock(
            user_query=f"user_query_{i}",
            assistant_final_text=f"assistant_response_{i}",
        )
        blocks.append(block)
    return blocks


class TestLibrarianCoreInit:
    """初始化测试"""

    def test_init_with_storage_only(self):
        """只有 storage 时正常初始化"""
        mock_storage = Mock()
        core = LibrarianCore(storage=mock_storage)
        assert core.storage is mock_storage
        assert core._bus is None
        assert core.generation_engine is None
        assert core.perception_layer is None

    def test_init_with_all_dependencies(self):
        """注入所有依赖时正常初始化"""
        mock_storage = Mock()
        mock_bus = Mock()
        mock_generation = Mock()
        mock_perception = Mock()
        mock_lifecycle = Mock()

        core = LibrarianCore(
            storage=mock_storage,
            bus=mock_bus,
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=mock_lifecycle,
        )

        assert core.storage is mock_storage
        assert core._bus is mock_bus
        assert core.generation_engine is mock_generation
        assert core.perception_layer is mock_perception
        assert core.lifecycle_engine is mock_lifecycle


class TestLibrarianCorePerceptionDelegation:
    """感知层代理接口测试"""

    def setup_method(self):
        self.perception = Mock()
        self.perception.manual_trigger = AsyncMock(return_value={
            "success": True,
            "topic_id": "topic_1",
            "message": "ok",
            "blocks_archived": 1,
        })
        self.perception.get_active_topics_snapshots = Mock(return_value=["snapshot"])
        self.core = LibrarianCore(storage=Mock(), perception_layer=self.perception)

    def test_get_active_topics_snapshots(self):
        identity = _make_identity()
        result = self.core.get_active_topics_snapshots(identity)
        assert result == ["snapshot"]
        self.perception.get_active_topics_snapshots.assert_called_once_with(identity)

    @pytest.mark.asyncio
    async def test_manual_trigger(self):
        result = await self.core.manual_trigger("topic_1")
        assert result["success"] is True
        self.perception.manual_trigger.assert_called_once_with("topic_1")

    @pytest.mark.asyncio
    async def test_manual_trigger_without_perception(self):
        core = LibrarianCore(storage=Mock(), perception_layer=None)
        result = await core.manual_trigger("topic_x")
        assert result["success"] is False


class TestLibrarianCoreGenerateMemory:
    """_on_generate_memory 回调测试"""

    def setup_method(self):
        self.mock_generation = MagicMock()
        self.mock_generation.process = AsyncMock(return_value=[])
        self.mock_storage = MagicMock()
        self.core = LibrarianCore(
            storage=self.mock_storage,
            generation_engine=self.mock_generation,
        )

    @pytest.mark.asyncio
    async def test_generate_memory_mode_a_default(self):
        """普通 flush，Phase 3: 构建 GenerationRequest(context=GenerationContext)"""
        from hivememory.engines.generation.models import GenerationContext
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="测试摘要",
            focus=None,
            reason=FlushReason.IDLE_TIMEOUT,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert isinstance(request, GenerationRequest)
        # Phase 4A: context 是 generation 唯一主字段
        assert request.context is not None
        assert isinstance(request.context, GenerationContext)
        assert len(request.context.turns) == 2
        assert request.context.state_summary == "测试摘要"
        assert request.write_focus is None
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_b_write(self):
        """MTP_WRITE flush，构建带 write_focus 的 request"""
        blocks = _make_logical_blocks(2)
        write_focus = WriteFocus(content="测试写入内容")
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=write_focus,
            reason=FlushReason.MTP_WRITE,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.write_focus is write_focus
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_b_write_without_context_still_runs(self):
        """MTP_WRITE 不应依赖上下文轮次，空背景也应进入 generation fallback"""
        write_focus = WriteFocus(content="测试写入内容")
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[],
            state_summary="",
            focus=write_focus,
            reason=FlushReason.MTP_WRITE,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.context is not None
        assert request.context.turns == []
        assert request.write_focus is write_focus
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_success(self):
        """MTP_UPDATE flush，加载 existing memory 成功"""
        blocks = _make_logical_blocks(2)
        # 使用真正的 UpdateFocus 实例而不是 Mock
        update_focus = UpdateFocus(
            instruction="更新测试",
            target_uuid=str(uuid4()),
            target_alias="fact_test",
            identity=_make_identity(),
        )
        existing_memory = Mock()

        # 设置 storage.get_memory 为异步 mock
        self.mock_storage.get_memory = AsyncMock(return_value=existing_memory)

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=update_focus,
            reason=FlushReason.MTP_UPDATE,
        )

        await self.core._on_generate_memory(payload)

        assert update_focus.existing_memory is existing_memory
        self.mock_generation.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_without_context_still_runs(self):
        """MTP_UPDATE 不应依赖上下文轮次，空背景也应进入 generation fallback"""
        update_focus = UpdateFocus(
            instruction="更新测试",
            target_uuid=str(uuid4()),
            target_alias="fact_test",
            identity=_make_identity(),
        )
        existing_memory = Mock()
        self.mock_storage.get_memory = AsyncMock(return_value=existing_memory)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[],
            state_summary="",
            focus=update_focus,
            reason=FlushReason.MTP_UPDATE,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.context is not None
        assert request.context.turns == []
        assert request.update_focus is update_focus
        assert update_focus.existing_memory is existing_memory

    @pytest.mark.asyncio
    async def test_generate_memory_mode_c_update_memory_not_found(self):
        """existing memory 不存在时 early return"""
        blocks = _make_logical_blocks(2)
        # 使用真正的 UpdateFocus 实例
        update_focus = UpdateFocus(
            instruction="更新测试",
            target_uuid=str(uuid4()),
            target_alias="fact_test",
            identity=_make_identity(),
        )

        # 设置 storage.get_memory 返回 None
        self.mock_storage.get_memory = AsyncMock(return_value=None)

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=update_focus,
            reason=FlushReason.MTP_UPDATE,
        )

        await self.core._on_generate_memory(payload)

        # generation.process 不应被调用
        self.mock_generation.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_memory_empty_blocks(self):
        """空 blocks 列表 early return"""
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[],
            state_summary="",
            focus=None,
            reason=FlushReason.IDLE_TIMEOUT,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_memory_blocks_with_only_user(self):
        """只有 user_query 的 blocks，Phase 3: 产出 1 个 turn（user_query 非空）"""
        block = LogicalBlock(user_query="user message", identity=_make_identity())
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=[block],
            state_summary="",
            focus=None,
            reason=FlushReason.IDLE_TIMEOUT,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        # Phase 3: context 字段包含 1 个 turn（有 user_query，被保留）
        assert request.context is not None
        assert len(request.context.turns) == 1

    @pytest.mark.asyncio
    async def test_generate_memory_generation_exception(self):
        """generation.process 抛异常时不崩溃"""
        blocks = _make_logical_blocks(2)
        self.mock_generation.process.side_effect = RuntimeError("generation failed")

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.IDLE_TIMEOUT,
        )

        # 不应抛异常
        await self.core._on_generate_memory(payload)

    @pytest.mark.asyncio
    async def test_generate_memory_without_generation_engine(self):
        """没有 generation_engine 时跳过处理"""
        core = LibrarianCore(storage=Mock())
        blocks = _make_logical_blocks(2)

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.IDLE_TIMEOUT,
        )

        # 不应抛异常
        await core._on_generate_memory(payload)

    @pytest.mark.asyncio
    async def test_generate_memory_semantic_drift(self):
        """SEMANTIC_DRIFT 触发 Mode A"""
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.SEMANTIC_DRIFT,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        assert request.write_focus is None
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_generate_memory_manual(self):
        """MANUAL 触发 Mode A"""
        blocks = _make_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.MANUAL,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_memory_kernel_blocks_with_topic_identity(self):
        identity = _make_identity()
        mock_perception = Mock()
        mock_perception.get_buffer.return_value = Mock(identity=identity)
        core = LibrarianCore(
            storage=self.mock_storage,
            generation_engine=self.mock_generation,
            perception_layer=mock_perception,
        )
        blocks = _make_kernel_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic-test-1",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.MANUAL,
        )

        await core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        # Phase 3: context 是主字段
        assert request.context is not None
        assert len(request.context.turns) == 2

    @pytest.mark.asyncio
    async def test_generate_memory_kernel_blocks_with_payload_identity(self):
        identity = _make_identity()
        blocks = _make_kernel_logical_blocks(2)
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=None,
            reason=FlushReason.MANUAL,
            identity=identity,
        )

        await self.core._on_generate_memory(payload)

        self.mock_generation.process.assert_called_once()
        request = self.mock_generation.process.call_args[0][0]
        # Phase 3: context 是主字段
        assert request.context is not None
        assert len(request.context.turns) == 2
