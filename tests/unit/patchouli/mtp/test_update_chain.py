"""
UPDATE 指令执行链路测试

验证 MTP UPDATE 指令从 Koakuma → LibrarianCore → GenerationEngine 的完整链路。

测试覆盖:
    1. UpdateFocus / MergeResult 数据模型
    2. GenerationRequest is_update 属性
    3. Mode C Merge Prompt 选择 (extractor)
    4. Mode C fallback 拼接
    5. _apply_update 版本历史追踪
    6. 双重处理防护 (MTP_UPDATE flush 不触发 Mode A)
    7. Koakuma._handle_update E2E
    8. Koakuma UPDATE 校验 (alias/instruction 缺失)

作者: HiveMemory Team
版本: 1.0
"""

import asyncio
import pytest
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from hivememory.core.models import (
    Identity, StreamMessage, StreamMessageType,
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, Artifacts, TurnRecord,
)
from hivememory.engines.generation.models import (
    UpdateFocus, MergeResult, GenerationRequest, GenerationContext, GenerationTurn, WriteFocus,
)
from hivememory.engines.perception.models import FlushReason, LogicalBlock, ArchivePayload
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.patchouli.services.librarian import LibrarianCore
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.system.config import KoakumaConfig
from hivememory.core.mtp import MTPResponseStatus


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
def existing_memory(identity) -> MemoryAtom:
    """模拟已存在的记忆 (UPDATE 的目标)"""
    return MemoryAtom(
        meta=MetaData(
            user_id=identity.user_id,
            source_agent_id=identity.agent_id,
            session_id=None,  # session_id 已从 Identity 中移除
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


def _execute_mtp(koakuma: KoakumaRuntime, text: str):
    return asyncio.run(koakuma.execute_mtp(text))


def _intercept_and_execute(koakuma: KoakumaRuntime, assistant_text: str):
    return asyncio.run(koakuma.intercept_and_execute(assistant_text))


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
        assert not req.is_write

    def test_is_write_with_write_focus(self):
        wf = WriteFocus(content="test")
        req = GenerationRequest(write_focus=wf)
        assert req.is_write
        assert not req.is_update

    def test_default_neither(self):
        req = GenerationRequest()
        assert not req.is_update
        assert not req.is_write

    def test_update_with_context(self, sample_context):
        uf = UpdateFocus(
            instruction="test", target_uuid="uuid-123", target_alias="alias",
        )
        req = GenerationRequest(context=sample_context, update_focus=uf)
        assert req.is_update
        assert len(req.context.turns) == 1


# ========== Test 4: Mode C Merge Prompt ==========

class TestModeCMergePrompt:
    """验证 Generation Engine Mode C 路径调用 extractor.merge()"""

    def test_mode_c_calls_merge_not_extract(self, identity, sample_context, existing_memory):
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

        request = GenerationRequest(context=sample_context, update_focus=uf)
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


# ========== Test 7: Double Processing Guard (UPDATE) ==========

import pytest
from hivememory.engines.perception.models import LogicalBlock
from hivememory.core.models import StreamMessage, StreamMessageType


class TestFlushCallbackModesUpdate:
    """验证 _on_generate_memory 统一回调的 UPDATE 模式分发"""

    @pytest.mark.asyncio
    async def test_mtp_update_flush_triggers_mode_c(self, sample_messages, existing_memory):
        """MTP_UPDATE flush 携带 update_focus → Mode C GenerationRequest"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []
        mock_storage = MagicMock()
        # 使用 AsyncMock 模拟异步方法
        mock_storage.get_memory = AsyncMock(return_value=existing_memory)

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_storage=mock_storage, mock_generation=mock_generation)
        core = LibrarianCore(
            storage=mock_storage,
            bus=bus,
            lifecycle_engine=MagicMock(),
            generation_engine=mock_generation,
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

        focus = UpdateFocus(
            instruction="把端口改成 9090",
            target_uuid=str(existing_memory.id),
            target_alias="fact_api_port",
            identity=Identity(user_id="test_user"),
        )

        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=focus,
            reason=FlushReason.MTP_UPDATE,
        )
        await core._on_generate_memory(payload)

        # generation_engine.process 应被调用，且携带 update_focus
        mock_generation.process.assert_called_once()
        request = mock_generation.process.call_args[0][0]
        assert request.update_focus is not None
        assert request.update_focus.instruction == "把端口改成 9090"
        assert request.write_focus is None
        # existing_memory 应被注入
        assert request.update_focus.existing_memory is existing_memory

    @pytest.mark.asyncio
    async def test_mtp_write_flush_also_triggers(self, sample_messages):
        """MTP_WRITE flush 携带 write_focus → Mode B"""
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            generation_engine=mock_generation,
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
        payload = ArchivePayload(
            topic_id="topic_test",
            blocks=blocks,
            state_summary="",
            focus=focus,
            reason=FlushReason.MTP_WRITE,
        )
        await core._on_generate_memory(payload)
        mock_generation.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_normal_flush_still_triggers_mode_a(self, sample_messages):
        mock_generation = MagicMock()
        mock_generation.process.return_value = []

        from .conftest import make_mock_bus
        bus = make_mock_bus(mock_generation=mock_generation)
        core = LibrarianCore(
            storage=MagicMock(),
            bus=bus,
            lifecycle_engine=MagicMock(),
            generation_engine=mock_generation,
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
            focus=None,
            reason=FlushReason.SEMANTIC_DRIFT,
        )
        await core._on_generate_memory(payload)
        mock_generation.process.assert_called_once()


# ========== Test 9: Koakuma UPDATE E2E ==========

class TestKoakumaUpdateE2E:
    """通过 MTP 指令验证 Koakuma UPDATE 完整链路"""

    @pytest.fixture
    def update_koakuma(self, existing_memory) -> KoakumaRuntime:
        mock_librarian = MagicMock()
        mock_librarian.handle_update_signal.return_value = [existing_memory]

        from .conftest import make_mock_bus
        bus = make_mock_bus()
        koakuma = KoakumaRuntime(bus=bus, config=KoakumaConfig())
        koakuma.set_current_identity(Identity(user_id="test_user"))

        # 注册 alias 到缓存
        koakuma.atom_cache.ingest_atom(existing_memory)
        return koakuma

    def test_update_basic(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="把端口改成 9090"'
        result = _intercept_and_execute(update_koakuma, agent_text)

        assert result is not None
        assert result.success

        # v3.0 延迟捕获: 验证 UpdateFocus 被暂存
        focus = update_koakuma.get_update_focus()
        assert focus is not None
        assert isinstance(focus, UpdateFocus)
        assert focus.instruction == "把端口改成 9090"
        assert focus.target_alias == "fact_api_port"

    def test_update_with_content(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="替换端口" content="port = 9090"'
        result = _intercept_and_execute(update_koakuma, agent_text)

        assert result is not None
        assert result.success

        focus = update_koakuma.get_update_focus()
        assert focus is not None
        assert focus.content == "port = 9090"
        assert focus.instruction == "替换端口"

    def test_update_response_contains_ack(self, update_koakuma):
        agent_text = '⟪ UPDATE | fact_api_port | instruction="test update"'
        result = _intercept_and_execute(update_koakuma, agent_text)

        assert result is not None
        assert "updated" in result.formatted_response.lower() or "ack" in result.formatted_response.lower()


# ========== Test 10: Koakuma UPDATE Validation ==========

class TestKoakumaUpdateValidation:
    """UPDATE 指令校验: alias 不存在、instruction 缺失等"""

    @pytest.fixture
    def validation_koakuma(self) -> KoakumaRuntime:
        from .conftest import make_mock_bus
        bus = make_mock_bus()
        koakuma = KoakumaRuntime(bus=bus, config=KoakumaConfig())
        koakuma.set_current_identity(Identity(user_id="test_user"))
        return koakuma

    def test_missing_instruction(self, validation_koakuma):
        # 注册 alias 但不提供 instruction
        validation_koakuma.atom_cache.ingest_atom(
            MemoryAtom(
                id=uuid4(),
                meta=MetaData(user_id="test_user", source_agent_id="test"),
                index=IndexLayer(
                    title="API Port Config",
                    summary="API port configuration fact",
                    memory_type=MemoryType.FACT,
                    alias="fact_api_port",
                ),
                payload=PayloadLayer(content="port = 8080"),
            )
        )
        agent_text = '⟪ UPDATE | fact_api_port | content="some content"'
        result = _intercept_and_execute(validation_koakuma, agent_text)

        assert result is not None
        assert "instruction" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        assert validation_koakuma.get_update_focus() is None

    def test_alias_not_found(self, validation_koakuma):
        agent_text = '⟪ UPDATE | nonexistent_alias | instruction="test"'
        result = _intercept_and_execute(validation_koakuma, agent_text)

        assert result is not None
        assert "not found" in result.formatted_response.lower() or "error" in result.formatted_response.lower()
        assert validation_koakuma.get_update_focus() is None

    def test_l2_route_failure_returns_infra_error(self, validation_koakuma):
        validation_koakuma._bus._mock_storage.get_memory_by_alias.side_effect = KeyError(
            "AsyncSystemBus: route 'storage.get_memory_by_alias' not registered"
        )
        agent_text = '⟪ UPDATE | fact_api_port | instruction="test"'
        result = _intercept_and_execute(validation_koakuma, agent_text)

        assert result is not None
        assert not result.success
        assert "Service Unavailable" in result.response_content
        assert validation_koakuma.get_update_focus() is None

    def test_update_deferred_capture_always_ack(self, existing_memory):
        """v3.0 延迟捕获: UPDATE 在 Koakuma 层始终返回 ACK"""
        from .conftest import make_mock_bus
        bus = make_mock_bus()
        koakuma = KoakumaRuntime(bus=bus, config=KoakumaConfig())
        koakuma.set_current_identity(Identity(user_id="test_user"))
        koakuma.atom_cache.ingest_atom(existing_memory)

        agent_text = '⟪ UPDATE | fact_api_port | instruction="test"'
        result = _intercept_and_execute(koakuma, agent_text)

        assert result is not None
        assert result.success
        assert koakuma.get_update_focus() is not None
        assert koakuma.get_update_focus().instruction == "test"


# ========== Test 11: FlushReason.MTP_UPDATE ==========

class TestFlushReasonMTPUpdate:
    """验证 FlushReason.MTP_UPDATE 枚举值"""

    def test_enum_value(self):
        assert FlushReason.MTP_UPDATE == "mtp_update"
        assert FlushReason.MTP_UPDATE.value == "mtp_update"

    def test_enum_member(self):
        assert hasattr(FlushReason, "MTP_UPDATE")
        assert FlushReason("mtp_update") == FlushReason.MTP_UPDATE
