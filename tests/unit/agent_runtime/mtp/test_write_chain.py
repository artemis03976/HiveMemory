"""
WRITE 指令执行链路测试

验证 MTP WRITE 指令从 Koakuma → LibrarianCore → GenerationEngine 的完整链路。

测试覆盖:
    1. WriteFocus / GenerationRequest 数据模型
    2. Mode B 提示词选择 (extractor)
    3. Mode B fallback 草稿构建
    4. 双重处理防护 (MTP_WRITE flush 不触发 Mode A)
    5. Koakuma._handle_write E2E

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, call, AsyncMock
from datetime import datetime

from hivememory.core.models import (
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
    StreamMessage,
    StreamMessageType,
    TurnRecord,
    WriteFocus,
)
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationRequest,
    GenerationContext,
    GenerationTurn,
    ExtractedMemoryDraft,
)
from hivememory.engines.perception.models import FlushReason, ArchivePayload
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.patchouli.services.librarian import LibrarianCore
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.system.config import KoakumaConfig
from hivememory.core.mtp import MTPResponseStatus
from hivememory.patchouli.control.memory_generation_tasks import MemoryGenerationTaskController


# ========== Fixtures ==========

@pytest.fixture
def identity() -> Identity:
    return Identity(user_id="test_user", agent_id="test_agent", session_id="test_session")


@pytest.fixture
def sample_messages(identity) -> list:
    return [
        StreamMessage(message_type=StreamMessageType.USER, content="帮我修复 CORS 问题", identity=identity),
        StreamMessage(message_type=StreamMessageType.ASSISTANT, content="已修复，端口从 8080 改为 9090", identity=identity),
    ]


@pytest.fixture
def sample_context(sample_messages, identity) -> GenerationContext:
    return GenerationContext(
        turns=[
            GenerationTurn(
                user_query=sample_messages[0].content,
                assistant_final_text=sample_messages[1].content,
                identity=identity,
            )
        ]
    )

@pytest.fixture
def sample_memory(identity) -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(user_id=identity.user_id, source_agent_id=identity.agent_id, session_id=None, confidence_score=1.0),
        index=IndexLayer(title="Fix CORS", summary="修复 CORS 跨域问题，端口从 8080 改为 9090", tags=["cors"], memory_type=MemoryType.FACT),
        payload=PayloadLayer(content="端口从 8080 改为 9090"),
    )


@pytest.fixture
def sample_draft() -> ExtractedMemoryDraft:
    return ExtractedMemoryDraft(
        title="Fix CORS",
        summary="端口改为 9090",
        tags=["cors", "nginx"],
        memory_type="FACT",
        content="端口从 8080 改为 9090",
        confidence_score=1.0,
        has_value=True,
        alias_suffix="fix_cors",
    )


async def _execute_mtp(koakuma: KoakumaRuntime, text: str, context=None):
    return await koakuma.execute_mtp(text, context=context)


async def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str, context=None):
    from .conftest import normalize_worker_agent_mtp_output

    return await koakuma.intercept_and_execute(
        normalize_worker_agent_mtp_output(assistant_text),
        context=context,
    )


# ========== Test 1: WriteFocus Model ==========

class TestWriteFocusModel:
    """WriteFocus 数据模型测试"""

    def test_basic_construction(self):
        focus = WriteFocus(content="def fix(): pass", reason="修复代码", title="Fix CORS")
        assert focus.content == "def fix(): pass"
        assert focus.reason == "修复代码"
        assert focus.title == "Fix CORS"

    def test_defaults(self):
        focus = WriteFocus(content="some content")
        assert focus.reason is None
        assert focus.title is None

    def test_dto_fields_only(self):
        focus = WriteFocus(content="test")
        assert focus.model_dump() == {
            "content": "test",
            "reason": None,
            "title": None,
        }

    def test_content_required(self):
        with pytest.raises(Exception):
            WriteFocus()


# ========== Test 2: GenerationRequest Model ==========

class TestGenerationRequest:
    """GenerationRequest 数据模型测试"""

    def test_mode_a_default(self, sample_context):
        req = GenerationRequest(context=sample_context)
        assert not req.is_write
        assert req.write_focus is None
        assert len(req.context.turns) == 1

    def test_mode_b_with_focus(self, sample_context):
        focus = WriteFocus(content="test content")
        req = GenerationRequest(context=sample_context, write_focus=focus)
        assert req.is_write
        assert req.write_focus.content == "test content"

    def test_empty_request(self):
        req = GenerationRequest()
        assert not req.is_write
        assert len(req.context.turns) == 0

    def test_focus_only_no_context(self):
        focus = WriteFocus(content="standalone write")
        req = GenerationRequest(write_focus=focus)
        assert req.is_write
        assert len(req.context.turns) == 0


# ========== Test 3: Engine Mode B Extraction ==========

class TestModeBExtraction:
    """验证 Generation Engine Mode B 路径"""

    @pytest.mark.asyncio
    async def test_mode_b_calls_extractor_with_write_metadata(self, identity, sample_context):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractedMemoryDraft(
            title="Fix CORS", summary="修复 CORS 跨域问题，端口从 8080 改为 9090", tags=["cors"],
            memory_type="FACT", content="端口改为 9090",
            confidence_score=1.0, has_value=True, alias_suffix="fix_cors",
        )
        mock_dedup = AsyncMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        focus = WriteFocus(content="端口改为 9090", reason="修复 CORS")
        request = GenerationRequest(context=sample_context, write_focus=focus)

        result = await engine.process(request=request)

        # 验证 extractor 被调用时 metadata 包含 mode=write
        call_args = mock_extractor.extract.call_args
        metadata = call_args[1]["metadata"] if "metadata" in call_args[1] else call_args[0][1]
        assert metadata["mode"] == "write"
        assert metadata["write_content"] == "端口改为 9090"
        assert metadata["write_reason"] == "修复 CORS"
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mode_a_no_write_metadata(self, sample_context):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractedMemoryDraft(
            title="Test Memory", summary="这是一条测试记忆，用于验证 Mode A 路径", tags=["test"],
            memory_type="FACT", content="test content for mode a",
            confidence_score=0.8, has_value=True, alias_suffix="test",
        )
        mock_dedup = AsyncMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        request = GenerationRequest(context=sample_context)
        result = await engine.process(request)

        call_args = mock_extractor.extract.call_args
        metadata = call_args[1]["metadata"] if "metadata" in call_args[1] else call_args[0][1]
        assert "mode" not in metadata or metadata.get("mode") != "write"


# ========== Test 4: Mode B Fallback ==========

class TestModeBFallback:
    """验证 LLM 提取失败时的 fallback 草稿构建"""

    @pytest.mark.asyncio
    async def test_fallback_when_extractor_returns_none(self, identity, sample_context):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None  # LLM 失败
        mock_dedup = AsyncMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        focus = WriteFocus(
            content="端口从 8080 改为 9090",
            reason="修复 CORS",
            title="Fix CORS Port",
        )
        request = GenerationRequest(context=sample_context, write_focus=focus)

        result = await engine.process(request=request)

        # fallback 应该保底生成 atom
        assert len(result) == 1
        assert result[0].atom is not None

    def test_fallback_draft_content(self):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        focus = WriteFocus(
            content="端口从 8080 改为 9090",
            reason="修复 CORS",
            title="Fix CORS Port",
        )
        draft = engine._build_fallback_draft(focus)

        assert draft.title == "Fix CORS Port"
        assert draft.content == "端口从 8080 改为 9090"
        assert draft.has_value is True
        assert draft.confidence_score == 1.0
        assert draft.memory_type == "FACT"
        assert "mtp_write" in draft.tags

    def test_fallback_draft_no_title(self):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        focus = WriteFocus(content="A very long content that should be truncated for title")
        draft = engine._build_fallback_draft(focus)
        assert draft.title == focus.content[:50]


# ========== Test 5: Double Processing Guard ==========

import pytest
from hivememory.engines.perception.models import LogicalBlock
from hivememory.core.models import StreamMessage, StreamMessageType


class TestFlushCallbackModes:
    """验证主动生成与被动归档已按新边界分离"""

    @pytest.mark.asyncio
    async def test_run_active_generation_triggers_mode_b(self, sample_messages):
        """主动 WRITE 由 run_active_generation 直驱 Mode B"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            perception_layer=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            task_controller=MemoryGenerationTaskController(storage=MagicMock(), generation_engine=mock_generation, bus=bus),
        )

        # 将 StreamMessage 转换为 LogicalBlock
        blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    identity=msg.identity,
                    user_query=msg.content,
                    assistant_final_text=msg.content if i % 2 == 1 else "",
                )
            )
            for i, msg in enumerate(sample_messages)
        ]

        focus = WriteFocus(content="端口改为 9090", reason="修复 CORS")
        core.perception_layer = MagicMock()
        core.perception_layer.get_topic_context.return_value = {
            "state_summary": "",
            "blocks": blocks,
        }
        task = PendingAtomMaterializeTask(
            pending_alias="draft_fix_cors_0001",
            intent_id="intent_fix_cors_0001",
            source_verb="WRITE",
            identity=Identity(user_id="test_user"),
            focus=focus,
        )
        memory_tasks = await core.run_active_generation([task], topic_id="topic_test")
        memory_task = memory_tasks[0]
        if memory_task._bg_task:
            await memory_task._bg_task

        mock_generation.process.assert_called_once()
        request = mock_generation.process.call_args[0][0]
        assert request.write_focus is not None
        assert request.write_focus.content == "端口改为 9090"
        assert request.update_focus is None

    @pytest.mark.asyncio
    async def test_run_active_generation_without_tasks_is_noop(self, sample_messages):
        """没有 materialize_tasks 时主动生成应直接跳过"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            perception_layer=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            task_controller=MemoryGenerationTaskController(storage=MagicMock(), generation_engine=mock_generation, bus=bus),
        )

        memory_tasks = await core.run_active_generation([], topic_id="topic_test")
        mock_generation.process.assert_not_called()
        assert memory_tasks == []

    @pytest.mark.asyncio
    async def test_normal_flush_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            perception_layer=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            task_controller=MemoryGenerationTaskController(storage=MagicMock(), generation_engine=mock_generation, bus=bus),
        )

        # 将 StreamMessage 转换为 LogicalBlock
        blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    identity=msg.identity,
                    user_query=msg.content,
                    assistant_final_text=msg.content if i % 2 == 1 else "",
                )
            )
            for i, msg in enumerate(sample_messages)
        ]

        # 普通被动 flush 应该触发 Mode A
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.MANUAL,
        )
        memory_task = await core._on_generate_memory(payload)
        if memory_task and memory_task._bg_task:
            await memory_task._bg_task

        mock_generation.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_flush_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            perception_layer=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            task_controller=MemoryGenerationTaskController(storage=MagicMock(), generation_engine=mock_generation, bus=bus),
        )

        # 将 StreamMessage 转换为 LogicalBlock
        blocks = [
            LogicalBlock(
                turn=TurnRecord(
                    identity=msg.identity,
                    user_query=msg.content,
                    assistant_final_text=msg.content if i % 2 == 1 else "",
                )
            )
            for i, msg in enumerate(sample_messages)
        ]

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            reason=FlushReason.MANUAL,
        )
        memory_task = await core._on_generate_memory(payload)
        if memory_task and memory_task._bg_task:
            await memory_task._bg_task
        mock_generation.process.assert_called_once()


# ========== Test 7: Koakuma WRITE E2E ==========

class TestKoakumaWriteE2E:
    """通过 MTP 指令验证 Koakuma WRITE 完整链路"""

    @pytest.fixture
    def write_koakuma(self, sample_memory) -> KoakumaRuntime:
        mock_librarian = MagicMock()
        mock_librarian.handle_write_signal.return_value = [sample_memory]

        from .conftest import make_koakuma_runtime, make_mock_bus
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        koakuma.context = MTPExecutionContext(identity=Identity(user_id="test_user"))
        return koakuma

    @pytest.mark.asyncio
    async def test_write_basic(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="端口从 8080 改为 9090" reason="修复 CORS"'
        result = await _intercept_and_execute(write_koakuma, agent_text, context=write_koakuma.context)

        assert result is not None
        assert result.success

        pending = write_koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        focus = pending.focus
        assert focus.content == "端口从 8080 改为 9090"
        assert focus.reason == "修复 CORS"
        assert pending.identity.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_write_with_title(self, write_koakuma):
        agent_text = '⟪ WRITE | * | title="Fix CORS" content="端口改为 9090" reason="修复"'
        result = await _intercept_and_execute(write_koakuma, agent_text, context=write_koakuma.context)

        assert result is not None
        pending = write_koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        focus = pending.focus
        assert focus.title == "Fix CORS"

    @pytest.mark.asyncio
    async def test_write_missing_content(self, write_koakuma):
        agent_text = '⟪ WRITE | * | reason="no content"'
        result = await _intercept_and_execute(write_koakuma, agent_text, context=write_koakuma.context)

        assert result is not None
        assert result.pending_alias is None

    @pytest.mark.asyncio
    async def test_write_response_contains_ack(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="test content"'
        result = await _intercept_and_execute(write_koakuma, agent_text, context=write_koakuma.context)

        assert result is not None
        assert result.pending_alias is not None
        assert "pending atom" in result.response_content
        assert result.pending_alias in result.response_content
        assert "ack" in result.formatted_response.lower()

    @pytest.mark.asyncio
    async def test_write_deferred_capture_always_ack(self):
        """v3.0 延迟捕获: WRITE 在 Koakuma 层始终返回 ACK，实际执行延迟到 payload 提交"""
        from .conftest import make_koakuma_runtime, make_mock_bus
        bus = make_mock_bus()
        koakuma = make_koakuma_runtime(bus, KoakumaConfig())
        from hivememory.core.models import RuntimeScope

        context = MTPExecutionContext(
            identity=Identity(user_id="test_user"),
            runtime_scope=RuntimeScope(
                run_id="run_write_test",
                frame_id="frame_main_write",
            ),
        )

        agent_text = '⟪ WRITE | * | content="test"'
        result = await _intercept_and_execute(koakuma, agent_text, context=context)

        assert result is not None
        assert result.success
        pending = koakuma.pending_runtime.get(result.pending_alias)
        assert pending is not None
        assert pending.focus.content == "test"
        assert pending.runtime_scope.run_id == "run_write_test"
        assert pending.runtime_scope.frame_id == "frame_main_write"


# ========== Test 8: Active Flush Reason Removed ==========

class TestFlushReasonActiveGenerationRemoved:
    """主动写生成已脱离感知层，不再保留 MTP flush reason"""

    def test_mtp_write_removed(self):
        assert "MTP_WRITE" not in FlushReason.__members__


# ========== Test 9: Engine unified API ==========

class TestEngineUnifiedAPI:
    """验证 process() 统一使用 GenerationRequest"""

    @pytest.mark.asyncio
    async def test_request_param_mode_a(self, sample_context):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None
        mock_dedup = AsyncMock()
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        request = GenerationRequest(context=sample_context)
        result = await engine.process(request)
        assert result == []
        mock_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_request_returns_empty(self):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=AsyncMock(),
        )
        result = await engine.process(GenerationRequest())
        assert result == []
