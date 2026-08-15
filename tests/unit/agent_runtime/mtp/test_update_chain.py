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

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from hivememory.core.models import (
    Identity, StreamMessage, StreamMessageType,
    MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType,
    UpdateFocus, WriteFocus,
)
from hivememory.engines.generation.models import (
    MergeResult, GenerationRequest, GenerationContext, GenerationTurn,
)
from hivememory.engines.perception.models import FlushReason
from hivememory.engines.generation.engine import MemoryGenerationEngine


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


def _mock_mid_term():
    mid_term = MagicMock()
    mid_term.search = AsyncMock(return_value=[])
    mid_term.upsert = AsyncMock()
    return mid_term


# ========== Test 3: GenerationRequest Update ==========

class TestGenerationRequestUpdate:
    """GenerationRequest is_update 属性测试"""

    def test_is_update_with_update_focus(self):
        uf = UpdateFocus(
            instruction="test", base_uuid="uuid-123", base_alias="alias",
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
            instruction="test", base_uuid="uuid-123", base_alias="alias",
        )
        req = GenerationRequest(context=sample_context, update_focus=uf)
        assert req.is_update
        assert len(req.context.turns) == 1


# ========== Test 4: Mode C Merge Prompt ==========

class TestModeCMergePrompt:
    """验证 Generation Engine Mode C 路径调用 extractor.merge()"""

    @pytest.mark.asyncio
    async def test_mode_c_calls_merge_not_extract(self, identity, sample_context, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="端口改为 9090", changelog="更新端口",
        )
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        old_content = existing_memory.payload.content  # 保存旧内容 (apply_update 会原地修改)

        uf = UpdateFocus(
            instruction="把端口改成 9090",
            base_uuid=str(existing_memory.id),
            base_alias="fact_api_port",
        )
        request = GenerationRequest(
            context=sample_context,
            update_focus=uf,
            existing_memory=existing_memory,
        )
        result = await engine.process(request=request)

        # merge() 被调用，extract() 不被调用
        mock_extractor.merge.assert_called_once()
        mock_extractor.extract.assert_not_called()

        # 验证 merge 参数
        call_args = mock_extractor.merge.call_args
        assert call_args[1]["old_content"] == old_content
        metadata = call_args[1]["metadata"]
        assert metadata["mode"] == "update"
        assert metadata["instruction"] == "把端口改成 9090"

    @pytest.mark.asyncio
    async def test_mode_c_returns_updated_memory(self, identity, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="新内容", changelog="测试更新",
        )
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="更新", base_uuid=str(existing_memory.id),
            base_alias="fact_api_port",
        )
        request = GenerationRequest(update_focus=uf, existing_memory=existing_memory)
        result = await engine.process(request=request)

        assert len(result) == 1
        assert result[0].atom.payload.content == "新内容"
        assert result[0].atom.get_alias() == existing_memory.get_alias()
        assert str(result[0].atom.id) == str(existing_memory.id)
        assert result[0].atom is not None


# ========== Test 5: Mode C Fallback ==========

class TestModeCFallback:
    """验证 LLM 合并失败时的 fallback 拼接"""

    @pytest.mark.asyncio
    async def test_fallback_when_merge_returns_none(self, identity, existing_memory):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = None  # LLM 失败
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="追加新内容",
            content="新增的段落",
            base_uuid=str(existing_memory.id),
            base_alias="fact_api_port",
        )
        request = GenerationRequest(update_focus=uf, existing_memory=existing_memory)
        result = await engine.process(request=request)

        # fallback 应该保底入库
        assert len(result) == 1
        assert result[0].atom is not None
        # fallback 拼接: 旧内容 + 新内容
        assert "新增的段落" in result[0].atom.payload.content
        assert existing_memory.payload.content.split("\n")[0] in result[0].atom.payload.content

    def test_fallback_content_append(self, existing_memory):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="追加内容",
            content="新段落文本",
            base_uuid="uuid-123",
            base_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory)

        assert isinstance(result, MergeResult)
        assert "新段落文本" in result.new_content
        assert existing_memory.payload.content in result.new_content
        assert "Fallback" in result.changelog

    def test_fallback_instruction_only(self, existing_memory):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="删除过时信息",
            content=None,
            base_uuid="uuid-123",
            base_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory)

        # 无 content 时保留旧内容不变
        assert result.new_content == existing_memory.payload.content
        assert "Fallback" in result.changelog
        assert "删除过时信息" in result.changelog

    @pytest.mark.asyncio
    async def test_mode_c_no_existing_memory_returns_empty(self, identity):
        """existing_memory 未注入时应返回空列表"""
        mock_extractor = MagicMock()
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage, extractor=mock_extractor, deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="test",
            base_uuid="uuid-123",
            base_alias="alias",
        )
        # 不注入 existing_memory (默认 None)

        request = GenerationRequest(update_focus=uf)
        result = await engine.process(request=request)

        assert result == []
        mock_extractor.merge.assert_not_called()


# ========== Test 6: _apply_update Version Tracking ==========

class TestApplyUpdate:
    """验证版本追踪 (history_summary, version++, changelog)"""

    def test_version_incremented(self, existing_memory, merge_result):
        mock_storage = _mock_mid_term()
        engine = MemoryGenerationEngine(
            mid_term=mock_storage, extractor=MagicMock(), deduplicator=MagicMock(),
        )

        old_version = existing_memory.meta.version
        result = engine._apply_update(existing_memory, merge_result)

        assert len(result) == 1
        assert result[0].atom.meta.version == old_version + 1

    def test_content_updated(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].atom.payload.content == merge_result.new_content

    def test_history_summary_appended(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        summary = result[0].atom.payload.history_summary
        assert len(summary) == 1
        assert merge_result.changelog in summary[0]
        assert result[0].changelog == merge_result.changelog

    def test_confidence_reset_to_1(self, existing_memory, merge_result):
        existing_memory.meta.confidence_score = 0.5
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].atom.meta.confidence_score == 1.0

    def test_updated_at_set(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )
        before = datetime.now()
        result = engine._apply_update(existing_memory, merge_result)

        assert result[0].atom.meta.updated_at >= before

    def test_apply_update_does_not_build_settlement(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )

        result = engine._apply_update(existing_memory, merge_result)

        assert not hasattr(result[0], "settlement")

    def test_multiple_updates_accumulate_history(self, existing_memory):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(), extractor=MagicMock(), deduplicator=MagicMock(),
        )

        # 第一次更新
        r1 = MergeResult(new_content="v2 content", changelog="first update")
        engine._apply_update(existing_memory, r1)

        # 第二次更新
        r2 = MergeResult(new_content="v3 content", changelog="second update")
        engine._apply_update(existing_memory, r2)

        assert existing_memory.meta.version == 3
        assert len(existing_memory.payload.history_summary) == 2


# ========== Test 7: Active UPDATE boundary ==========


class TestActiveUpdateBoundary:
    """Active UPDATE generation is no longer represented as a perception flush."""

    def test_mtp_update_flush_reason_removed(self):
        assert "MTP_UPDATE" not in FlushReason.__members__

    def test_update_focus_stays_on_generation_request(self):
        focus = UpdateFocus(
            instruction="change port",
            base_uuid="uuid-123",
            base_alias="fact_api_port",
        )
        request = GenerationRequest(update_focus=focus)

        assert request.is_update
        assert request.update_focus is focus
        assert request.write_focus is None


# ========== Test 11: Active Flush Reason Removed ==========

class TestFlushReasonActiveGenerationRemoved:
    """主动更新生成已脱离感知层，不再保留 MTP flush reason"""

    def test_mtp_update_removed(self):
        assert "MTP_UPDATE" not in FlushReason.__members__
