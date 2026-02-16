"""
UPDATE 指令执行链路测试

验证 MTP UPDATE 指令从 Koakuma → LibrarianCore → GenerationEngine 的完整链路。

测试覆盖:
    1. UpdateFocus / MergeResult 数据模型
    2. GenerationRequest is_update 属性
    3. Mode C Merge Prompt 选择 (extractor)
    4. Mode C fallback 拼接
    5. _apply_update 版本历史追踪
    6. LibrarianCore.handle_update_signal 流程
    7. 双重处理防护 (MTP_UPDATE flush 不触发 Mode A)
    8. Koakuma._handle_update E2E
    9. Koakuma UPDATE 校验 (alias/instruction 缺失)

作者: HiveMemory Team
版本: 1.0
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from uuid import uuid4

from hivememory.core.models import (
    Identity, StreamMessage, StreamMessageType,
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, Artifacts,
)
from hivememory.engines.generation.models import (
    UpdateFocus, MergeResult, GenerationRequest, WriteFocus,
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
        StreamMessage(message_type=StreamMessageType.USER, content="帮我把 API 端口改成 9090", identity=identity),
        StreamMessage(message_type=StreamMessageType.ASSISTANT, content="好的，已修改端口配置", identity=identity),
    ]


@pytest.fixture
def existing_memory(identity) -> MemoryAtom:
    """模拟已存在的记忆 (UPDATE 的目标)"""
    return MemoryAtom(
        meta=MetaData(
            user_id=identity.user_id,
            source_agent_id=identity.agent_id,
            session_id=identity.session_id,
            confidence_score=0.85,
            version=1,
        ),
        index=IndexLayer(
            title="API 端口配置",
            summary="API 服务端口为 8080",
            tags=["api", "config"],
            memory_type=MemoryType.FACT,
            alias="fact_api_port",
        ),
        payload=PayloadLayer(
            content="API 服务运行在端口 8080，使用 HTTP 协议。",
        ),
    )


@pytest.fixture
def merge_result() -> MergeResult:
    return MergeResult(
        new_content="API 服务运行在端口 9090，使用 HTTP 协议。",
        changelog="端口从 8080 更新为 9090",
    )


# ========== Test 1: UpdateFocus Model ==========

class TestUpdateFocusModel:
    """UpdateFocus 数据模型测试"""

    def test_basic_construction(self):
        focus = UpdateFocus(
            instruction="把端口改成 9090",
            content="port = 9090",
            target_uuid="uuid-123",
            target_alias="fact_api_port",
        )
        assert focus.instruction == "把端口改成 9090"
        assert focus.content == "port = 9090"
        assert focus.target_uuid == "uuid-123"
        assert focus.target_alias == "fact_api_port"

    def test_defaults(self):
        focus = UpdateFocus(
            instruction="修改端口",
            target_uuid="uuid-123",
            target_alias="fact_api_port",
        )
        assert focus.content is None
        assert focus.existing_memory is None
        assert focus.identity is not None

    def test_with_identity(self, identity):
        focus = UpdateFocus(
            instruction="test",
            target_uuid="uuid-123",
            target_alias="alias",
            identity=identity,
        )
        assert focus.identity.user_id == "test_user"
        assert focus.identity.agent_id == "test_agent"

    def test_instruction_required(self):
        with pytest.raises(Exception):
            UpdateFocus(target_uuid="uuid-123", target_alias="alias")

    def test_target_uuid_required(self):
        with pytest.raises(Exception):
            UpdateFocus(instruction="test", target_alias="alias")

    def test_existing_memory_injection(self, existing_memory):
        focus = UpdateFocus(
            instruction="test",
            target_uuid="uuid-123",
            target_alias="alias",
        )
        focus.existing_memory = existing_memory
        assert focus.existing_memory.index.title == "API 端口配置"


# ========== Test 2: MergeResult Model ==========

class TestMergeResult:
    """MergeResult 数据模型测试"""

    def test_basic_construction(self):
        result = MergeResult(new_content="new stuff", changelog="updated content")
        assert result.new_content == "new stuff"
        assert result.changelog == "updated content"

    def test_serialization(self):
        result = MergeResult(new_content="content", changelog="log")
        data = result.model_dump()
        assert data["new_content"] == "content"
        assert data["changelog"] == "log"

    def test_required_fields(self):
        with pytest.raises(Exception):
            MergeResult(new_content="only content")
        with pytest.raises(Exception):
            MergeResult(changelog="only log")


# ========== Test 3: GenerationRequest Update ==========

class TestGenerationRequestUpdate:
    """GenerationRequest is_update 属性测试"""

    def test_is_update_with_update_focus(self):
        uf = UpdateFocus(
            instruction="test", target_uuid="uuid-123", target_alias="alias",
        )
        req = GenerationRequest(update_focus=uf)
        assert req.is_update
        assert not req.is_focused

    def test_is_focused_with_write_focus(self):
        wf = WriteFocus(content="test")
        req = GenerationRequest(write_focus=wf)
        assert req.is_focused
        assert not req.is_update

    def test_default_neither(self):
        req = GenerationRequest()
        assert not req.is_update
        assert not req.is_focused

    def test_update_with_context(self, sample_messages):
        uf = UpdateFocus(
            instruction="test", target_uuid="uuid-123", target_alias="alias",
        )
        req = GenerationRequest(context_messages=sample_messages, update_focus=uf)
        assert req.is_update
        assert len(req.context_messages) == 2


# ========== Test 4: Mode C Merge Prompt ==========

class TestModeCMergePrompt:
    """验证 Generation Engine Mode C 路径调用 extractor.merge()"""

    def test_mode_c_calls_merge_not_extract(self, identity, sample_messages, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="端口改为 9090", changelog="更新端口",
        )
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        old_content = existing_memory.payload.content  # 保存旧内容 (apply_update 会原地修改)

        uf = UpdateFocus(
            instruction="把端口改成 9090",
            target_uuid=str(existing_memory.id),
            target_alias="fact_api_port",
            identity=identity,
        )
        uf.existing_memory = existing_memory

        request = GenerationRequest(context_messages=sample_messages, update_focus=uf)
        result = engine.process(request=request)

        # merge() 被调用，extract() 不被调用
        mock_extractor.merge.assert_called_once()
        mock_extractor.extract.assert_not_called()

        # 验证 merge 参数
        call_args = mock_extractor.merge.call_args
        assert call_args[1]["old_content"] == old_content
        metadata = call_args[1]["metadata"]
        assert metadata["mode"] == "update"
        assert metadata["instruction"] == "把端口改成 9090"

    def test_mode_c_returns_updated_memory(self, identity, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="新内容", changelog="测试更新",
        )
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="更新", target_uuid=str(existing_memory.id),
            target_alias="fact_api_port", identity=identity,
        )
        uf.existing_memory = existing_memory

        request = GenerationRequest(update_focus=uf)
        result = engine.process(request=request)

        assert len(result) == 1
        assert result[0].payload.content == "新内容"
        mock_storage.upsert_memory.assert_called_once()


# ========== Test 5: Mode C Fallback ==========

class TestModeCFallback:
    """验证 LLM 合并失败时的 fallback 拼接"""

    def test_fallback_when_merge_returns_none(self, identity, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = None  # LLM 失败
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="追加新内容",
            content="新增的段落",
            target_uuid=str(existing_memory.id),
            target_alias="fact_api_port",
            identity=identity,
        )
        uf.existing_memory = existing_memory

        request = GenerationRequest(update_focus=uf)
        result = engine.process(request=request)

        # fallback 应该保底入库
        assert len(result) == 1
        assert mock_storage.upsert_memory.called
        # fallback 拼接: 旧内容 + 新内容
        assert "新增的段落" in result[0].payload.content
        assert existing_memory.payload.content.split("\n")[0] in result[0].payload.content

    def test_fallback_content_append(self, existing_memory):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="追加内容",
            content="新段落文本",
            target_uuid="uuid-123",
            target_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory)

        assert isinstance(result, MergeResult)
        assert "新段落文本" in result.new_content
        assert existing_memory.payload.content in result.new_content
        assert "Fallback" in result.changelog

    def test_fallback_instruction_only(self, existing_memory):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="删除过时信息",
            content=None,
            target_uuid="uuid-123",
            target_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory)

        # 无 content 时保留旧内容不变
        assert result.new_content == existing_memory.payload.content
        assert "Fallback" in result.changelog
        assert "删除过时信息" in result.changelog

    def test_mode_c_no_existing_memory_returns_empty(self, identity):
        """existing_memory 未注入时应返回空列表"""
        mock_extractor = MagicMock()
        mock_storage = MagicMock()

        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="test",
            target_uuid="uuid-123",
            target_alias="alias",
            identity=identity,
        )
        # 不注入 existing_memory (默认 None)

        request = GenerationRequest(update_focus=uf)
        result = engine.process(request=request)

        assert result == []
        mock_extractor.merge.assert_not_called()


# ========== Test 6: _apply_update Version History ==========

class TestApplyUpdate:
    """验证版本历史追踪 (full_history, history_summary, version++)"""

    def test_version_incremented(self, existing_memory, merge_result):
        mock_storage = MagicMock()
        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=MagicMock(), deduplicator=MagicMock(),
        )

        old_version = existing_memory.meta.version
        result = engine._apply_update(existing_memory, merge_result)

        assert len(result) == 1
        assert result[0].meta.version == old_version + 1

    def test_content_updated(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].payload.content == merge_result.new_content

    def test_full_history_pushed(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        old_content = existing_memory.payload.content

        result = engine._apply_update(existing_memory, merge_result)

        history = result[0].payload.artifacts.full_history
        assert len(history) == 1
        assert history[0]["content"] == old_content
        assert history[0]["reason"] == merge_result.changelog
        assert "timestamp" in history[0]

    def test_history_summary_appended(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        summary = result[0].payload.history_summary
        assert len(summary) == 1
        assert merge_result.changelog in summary[0]

    def test_confidence_reset_to_1(self, existing_memory, merge_result):
        existing_memory.meta.confidence_score = 0.5
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].meta.confidence_score == 1.0

    def test_updated_at_set(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        before = datetime.now()
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].meta.updated_at >= before

    def test_persisted_to_storage(self, existing_memory, merge_result):
        mock_storage = MagicMock()
        engine = MemoryGenerationEngine(
            storage=mock_storage, extractor=MagicMock(), deduplicator=MagicMock(),
        )
        engine._apply_update(existing_memory, merge_result)

        mock_storage.upsert_memory.assert_called_once_with(existing_memory)

    def test_multiple_updates_accumulate_history(self, existing_memory):
        engine = MemoryGenerationEngine(
            storage=MagicMock(), extractor=MagicMock(), deduplicator=MagicMock(),
        )

        # 第一次更新
        r1 = MergeResult(new_content="v2 content", changelog="first update")
        engine._apply_update(existing_memory, r1)

        # 第二次更新
        r2 = MergeResult(new_content="v3 content", changelog="second update")
        engine._apply_update(existing_memory, r2)

        assert existing_memory.meta.version == 3
        assert len(existing_memory.payload.artifacts.full_history) == 2
        assert len(existing_memory.payload.history_summary) == 2


# ========== Test 7: LibrarianCore.handle_update_signal ==========

class TestHandleUpdateSignal:
    """验证 LibrarianCore UPDATE 信号处理流程"""

    def test_flush_load_and_generate(self, identity, sample_messages, existing_memory):
        mock_perception = MagicMock()
        mock_perception.flush_buffer.return_value = sample_messages

        mock_generation = MagicMock()
        mock_generation.process.return_value = [existing_memory]

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = existing_memory

        core = LibrarianCore(
            storage=mock_storage,
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="更新端口",
            target_uuid=str(existing_memory.id),
            target_alias="fact_api_port",
            identity=identity,
        )
        result = core.handle_update_signal(uf)

        # 验证 flush_buffer 被调用 (reason=MTP_UPDATE)
        mock_perception.flush_buffer.assert_called_once_with(
            identity=identity,
            reason=FlushReason.MTP_UPDATE,
        )

        # 验证 storage.get_memory 被调用
        mock_storage.get_memory.assert_called_once()

        # 验证 generation_engine.process 被调用 (Mode C request)
        call_args = mock_generation.process.call_args
        request = call_args[0][0]
        assert isinstance(request, GenerationRequest)
        assert request.is_update
        assert request.update_focus.instruction == "更新端口"
        assert request.update_focus.existing_memory == existing_memory
        assert request.context_messages == sample_messages

        assert len(result) == 1

    def test_memory_not_found_raises(self, identity):
        mock_perception = MagicMock()
        mock_perception.flush_buffer.return_value = []

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = None  # 记忆不存在

        core = LibrarianCore(
            storage=mock_storage,
            generation_engine=MagicMock(),
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="test",
            target_uuid=str(uuid4()),
            target_alias="nonexistent",
            identity=identity,
        )

        with pytest.raises(ValueError, match="not found in storage"):
            core.handle_update_signal(uf)

    def test_generation_failure_returns_empty(self, identity, existing_memory):
        mock_perception = MagicMock()
        mock_perception.flush_buffer.return_value = []

        mock_generation = MagicMock()
        mock_generation.process.side_effect = Exception("LLM error")

        mock_storage = MagicMock()
        mock_storage.get_memory.return_value = existing_memory

        core = LibrarianCore(
            storage=mock_storage,
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="test",
            target_uuid=str(existing_memory.id),
            target_alias="fact_api_port",
            identity=identity,
        )
        result = core.handle_update_signal(uf)

        assert result == []


# ========== Test 8: Double Processing Guard (UPDATE) ==========

class TestDoubleProcessingGuardUpdate:
    """验证 MTP_UPDATE flush 不触发 Mode A 回调"""

    def test_mtp_update_flush_skips_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_perception = MagicMock()

        core = LibrarianCore(
            storage=MagicMock(),
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        # 模拟 MTP_UPDATE 原因的 flush 回调
        core._on_perception_flush(sample_messages, FlushReason.MTP_UPDATE)

        # generation_engine.process 不应被调用 (Mode A 被跳过)
        mock_generation.process.assert_not_called()

    def test_mtp_write_flush_also_skips(self, sample_messages):
        mock_generation = MagicMock()
        mock_perception = MagicMock()

        core = LibrarianCore(
            storage=MagicMock(),
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        core._on_perception_flush(sample_messages, FlushReason.MTP_WRITE)
        mock_generation.process.assert_not_called()

    def test_normal_flush_still_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []
        mock_perception = MagicMock()

        core = LibrarianCore(
            storage=MagicMock(),
            generation_engine=mock_generation,
            perception_layer=mock_perception,
            lifecycle_engine=MagicMock(),
        )

        core._on_perception_flush(sample_messages, FlushReason.SEMANTIC_DRIFT)
        mock_generation.process.assert_called_once()


# ========== Test 9: Koakuma UPDATE E2E ==========

class TestKoakumaUpdateE2E:
    """通过 MTP 指令验证 Koakuma UPDATE 完整链路"""

    @pytest.fixture
    def update_koakuma(self, existing_memory) -> KoakumaRuntime:
        mock_librarian = MagicMock()
        mock_librarian.handle_update_signal.return_value = [existing_memory]

        mock_storage = MagicMock()

        koakuma = KoakumaRuntime(
            retrieval_familiar=MagicMock(),
            librarian_core=mock_librarian,
            storage=mock_storage,
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")

        # 注册 alias 到 resolver
        koakuma.alias_resolver.register_context_alias(
            "fact_api_port", str(existing_memory.id),
        )
        return koakuma

    def test_update_basic(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="把端口改成 9090"'
        result = update_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success

        mock_librarian = update_koakuma._librarian
        mock_librarian.handle_update_signal.assert_called_once()

        call_args = mock_librarian.handle_update_signal.call_args[0][0]
        assert isinstance(call_args, UpdateFocus)
        assert call_args.instruction == "把端口改成 9090"
        assert call_args.target_alias == "fact_api_port"

    def test_update_with_content(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="替换端口" content="port = 9090"'
        result = update_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert result.success

        call_args = update_koakuma._librarian.handle_update_signal.call_args[0][0]
        assert call_args.content == "port = 9090"
        assert call_args.instruction == "替换端口"

    def test_update_response_contains_ack(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="test update"'
        result = update_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert "updated" in result.formatted_response.lower() or "ack" in result.formatted_response.lower()


# ========== Test 10: Koakuma UPDATE Validation ==========

class TestKoakumaUpdateValidation:
    """UPDATE 指令校验: alias 不存在、instruction 缺失等"""

    @pytest.fixture
    def validation_koakuma(self) -> KoakumaRuntime:
        koakuma = KoakumaRuntime(
            retrieval_familiar=MagicMock(),
            librarian_core=MagicMock(),
            storage=MagicMock(),
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        return koakuma

    def test_missing_instruction(self, validation_koakuma):
        # 注册 alias 但不提供 instruction
        validation_koakuma.alias_resolver.register_context_alias("fact_api_port", "uuid-123")
        agent_text = '⟪ UPDATE | fact_api_port | content="some content"'
        result = validation_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert "instruction" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        validation_koakuma._librarian.handle_update_signal.assert_not_called()

    def test_alias_not_found(self, validation_koakuma):
        agent_text = '⟪ UPDATE | nonexistent_alias | instruction="test"'
        result = validation_koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert "not found" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        validation_koakuma._librarian.handle_update_signal.assert_not_called()

    def test_update_failure_returns_error(self, existing_memory):
        mock_librarian = MagicMock()
        mock_librarian.handle_update_signal.side_effect = Exception("DB error")

        koakuma = KoakumaRuntime(
            retrieval_familiar=MagicMock(),
            librarian_core=mock_librarian,
            storage=MagicMock(),
            config=KoakumaConfig(),
        )
        koakuma.set_current_user("test_user")
        koakuma.alias_resolver.register_context_alias(
            "fact_api_port", str(existing_memory.id),
        )

        agent_text = '⟪ UPDATE | fact_api_port | instruction="test"'
        result = koakuma.intercept_and_execute(agent_text)

        assert result is not None
        assert "error" in result.formatted_response.lower() or "fail" in result.formatted_response.lower()


# ========== Test 11: FlushReason.MTP_UPDATE ==========

class TestFlushReasonMTPUpdate:
    """验证 FlushReason.MTP_UPDATE 枚举值"""

    def test_enum_value(self):
        assert FlushReason.MTP_UPDATE == "mtp_update"
        assert FlushReason.MTP_UPDATE.value == "mtp_update"

    def test_enum_member(self):
        assert hasattr(FlushReason, "MTP_UPDATE")
        assert FlushReason("mtp_update") == FlushReason.MTP_UPDATE

