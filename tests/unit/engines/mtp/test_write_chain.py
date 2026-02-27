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

import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from hivememory.core.models import Identity, StreamMessage, StreamMessageType, MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.engines.generation.models import (
    WriteFocus,
    GenerationRequest,
    ExtractedMemoryDraft,
    DuplicateDecision,
)
from hivememory.engines.perception.models import FlushReason
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.patchouli.kernel.librarian_core import LibrarianCore
from hivememory.patchouli.kernel.koakuma import KoakumaRuntime
from hivememory.patchouli.config import KoakumaConfig
from hivememory.patchouli.protocol.mtp import MTPResponseStatus


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
def sample_memory(identity) -> MemoryAtom:
    return MemoryAtom(
        meta=MetaData(user_id=identity.user_id, source_agent_id=identity.agent_id, session_id=identity.session_id, confidence_score=1.0),
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
        assert focus.identity is not None

    def test_with_identity(self, identity):
        focus = WriteFocus(content="test", identity=identity)
        assert focus.identity.user_id == "test_user"
        assert focus.identity.agent_id == "test_agent"

    def test_content_required(self):
        with pytest.raises(Exception):
            WriteFocus()


# ========== Test 2: GenerationRequest Model ==========

class TestGenerationRequest:
    """GenerationRequest 数据模型测试"""

    def test_mode_a_default(self, sample_messages):
        req = GenerationRequest(context_messages=sample_messages)
        assert not req.is_focused
        assert req.write_focus is None
        assert len(req.context_messages) == 2

    def test_mode_b_with_focus(self, sample_messages):
        focus = WriteFocus(content="test content")
        req = GenerationRequest(context_messages=sample_messages, write_focus=focus)
        assert req.is_focused
        assert req.write_focus.content == "test content"

    def test_empty_request(self):
        req = GenerationRequest()
        assert not req.is_focused
        assert len(req.context_messages) == 0

    def test_focus_only_no_context(self):
        focus = WriteFocus(content="standalone write")
        req = GenerationRequest(write_focus=focus)
        assert req.is_focused
        assert len(req.context_messages) == 0


# ========== Test 3: Engine Mode B Extraction ==========

class TestModeBExtraction:
    """验证 Generation Engine Mode B 路径"""

    def test_mode_b_calls_extractor_with_write_metadata(self, identity, sample_messages):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractedMemoryDraft(
            title="Fix CORS", summary="修复 CORS 跨域问题，端口从 8080 改为 9090", tags=["cors"],
            memory_type="FACT", content="端口改为 9090",
            confidence_score=1.0, has_value=True, alias_suffix="fix_cors",
        )
        mock_dedup = MagicMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        focus = WriteFocus(content="端口改为 9090", reason="修复 CORS", identity=identity)
        request = GenerationRequest(context_messages=sample_messages, write_focus=focus)

        result = engine.process(request=request)

        # 验证 extractor 被调用时 metadata 包含 mode=write
        call_args = mock_extractor.extract.call_args
        metadata = call_args[1]["metadata"] if "metadata" in call_args[1] else call_args[0][1]
        assert metadata["mode"] == "write"
        assert metadata["write_content"] == "端口改为 9090"
        assert metadata["write_reason"] == "修复 CORS"
        assert len(result) == 1

    def test_mode_a_no_write_metadata(self, sample_messages):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ExtractedMemoryDraft(
            title="Test Memory", summary="这是一条测试记忆，用于验证 Mode A 路径", tags=["test"],
            memory_type="FACT", content="test content for mode a",
            confidence_score=0.8, has_value=True, alias_suffix="test",
        )
        mock_dedup = MagicMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        request = GenerationRequest(context_messages=sample_messages)
        result = engine.process(request)

        call_args = mock_extractor.extract.call_args
        metadata = call_args[1]["metadata"] if "metadata" in call_args[1] else call_args[0][1]
        assert "mode" not in metadata or metadata.get("mode") != "write"


# ========== Test 4: Mode B Fallback ==========

class TestModeBFallback:
    """验证 LLM 提取失败时的 fallback 草稿构建"""

    def test_fallback_when_extractor_returns_none(self, identity, sample_messages):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None  # LLM 失败
        mock_dedup = MagicMock()
        mock_dedup.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        focus = WriteFocus(
            content="端口从 8080 改为 9090",
            reason="修复 CORS",
            title="Fix CORS Port",
            identity=identity,
        )
        request = GenerationRequest(context_messages=sample_messages, write_focus=focus)

        result = engine.process(request=request)

        # fallback 应该保底入库
        assert len(result) == 1
        assert mock_storage.upsert_memory.called

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

class TestFlushCallbackModes:
    """验证 _on_perception_flush 统一回调的模式分发"""

    def test_mtp_write_flush_triggers_mode_b(self, sample_messages):
        """MTP_WRITE flush 携带 write_focus → Mode B GenerationRequest"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
        )

        focus = WriteFocus(content="端口改为 9090", reason="修复 CORS")
        core._on_perception_flush(
            sample_messages, FlushReason.MTP_WRITE,
            write_focus=focus,
        )

        # generation_engine.process 应被调用，且携带 write_focus
        mock_generation.process.assert_called_once()
        request = mock_generation.process.call_args[0][0]
        assert request.write_focus is not None
        assert request.write_focus.content == "端口改为 9090"
        assert request.update_focus is None

    def test_mtp_write_flush_without_focus_triggers_mode_a(self, sample_messages):
        """MTP_WRITE flush 但无 write_focus → 降级为 Mode A"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
        )

        core._on_perception_flush(sample_messages, FlushReason.MTP_WRITE)

        mock_generation.process.assert_called_once()
        request = mock_generation.process.call_args[0][0]
        assert request.write_focus is None

    def test_normal_flush_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
        )

        # 正常 flush 应该触发 Mode A
        core._on_perception_flush(sample_messages, FlushReason.SEMANTIC_DRIFT)

        mock_generation.process.assert_called_once()

    def test_manual_flush_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
        )

        core._on_perception_flush(sample_messages, FlushReason.MANUAL)
        mock_generation.process.assert_called_once()


# ========== Test 7: Koakuma WRITE E2E ==========

class TestKoakumaWriteE2E:
    """通过 MTP 指令验证 Koakuma WRITE 完整链路"""

    @pytest.fixture
    def write_koakuma(self, sample_memory) -> KoakumaRuntime:
        mock_librarian = MagicMock()
        mock_librarian.handle_write_signal.return_value = [sample_memory]

        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus()
        koakuma = KoakumaRuntime(bus=bus, config=KoakumaConfig())
        koakuma.set_current_user("test_user")
        return koakuma

    def test_write_basic(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="端口从 8080 改为 9090" reason="修复 CORS"'
        result = write_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success

        # v3.0 延迟捕获: 验证 WriteFocus 被暂存而非直接调用 librarian
        focus = write_koakuma.get_write_focus()
        assert focus is not None
        assert focus.content == "端口从 8080 改为 9090"
        assert focus.reason == "修复 CORS"
        assert focus.identity.user_id == "test_user"

    def test_write_with_title(self, write_koakuma):
        agent_text = '⟪ WRITE | * | title="Fix CORS" content="端口改为 9090" reason="修复"'
        result = write_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        focus = write_koakuma.get_write_focus()
        assert focus is not None
        assert focus.title == "Fix CORS"

    def test_write_missing_content(self, write_koakuma):
        agent_text = '⟪ WRITE | * | reason="no content"'
        result = write_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        # 应该返回 error，不捕获 WriteFocus
        assert write_koakuma.get_write_focus() is None

    def test_write_response_contains_ack(self, write_koakuma):
        agent_text = '⟪ WRITE | * | content="test content"'
        result = write_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        # 响应应包含 status=ack
        assert "ack" in result.formatted_response.lower() or "saved" in result.formatted_response.lower()

    def test_write_deferred_capture_always_ack(self):
        """v3.0 延迟捕获: WRITE 在 Koakuma 层始终返回 ACK，实际执行延迟到 payload 提交"""
        from tests.unit.engines.mtp.conftest import make_mock_bus
        bus = make_mock_bus()
        koakuma = KoakumaRuntime(bus=bus, config=KoakumaConfig())
        koakuma.set_current_user("test_user")

        agent_text = '⟪ WRITE | * | content="test"'
        result = koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success
        assert koakuma.get_write_focus() is not None
        assert koakuma.get_write_focus().content == "test"


# ========== Test 8: FlushReason.MTP_WRITE ==========

class TestFlushReasonMTPWrite:
    """验证 FlushReason.MTP_WRITE 枚举值"""

    def test_enum_value(self):
        assert FlushReason.MTP_WRITE == "mtp_write"
        assert FlushReason.MTP_WRITE.value == "mtp_write"

    def test_enum_member(self):
        assert hasattr(FlushReason, "MTP_WRITE")
        assert FlushReason("mtp_write") == FlushReason.MTP_WRITE


# ========== Test 9: Engine unified API ==========

class TestEngineUnifiedAPI:
    """验证 process() 统一使用 GenerationRequest"""

    def test_request_param_mode_a(self, sample_messages):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = None
        mock_dedup = MagicMock()
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=mock_dedup,
        )

        request = GenerationRequest(context_messages=sample_messages)
        result = engine.process(request)
        assert result == []
        mock_extractor.extract.assert_called_once()

    def test_empty_request_returns_empty(self):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine.process(GenerationRequest())
        assert result == []
