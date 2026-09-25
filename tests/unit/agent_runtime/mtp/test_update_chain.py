"""
UPDATE 指令执行链路测试

验证 MTP UPDATE 指令从 Koakuma → LibrarianCore → GenerationEngine 的完整链路。

测试覆盖:
    1. UpdateFocus / MergeResult 数据模型
    2. GenerationRequest is_update 属性
    3. Mode C Merge Prompt 选择 (extractor)
    4. Mode C fallback 拼接
    5. _apply_update 纯计算边界（内容合并 + before 快照；版本/时间归 Familiar）
    6. 双重处理防护 (MTP_UPDATE flush 不触发 Mode A)
    7. Koakuma._handle_update E2E
    8. Koakuma UPDATE 校验 (alias/instruction 缺失)

作者: HiveMemory Team
版本: 1.0
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    StreamMessage,
    StreamMessageType,
    UpdateFocus,
)
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
    MergeResult,
    system_settlement_provenance,
)
from hivememory.engines.perception.models import TriggerReason
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata

# ========== Fixtures ==========

# 提交边界固定时点：引擎层验证传入 now 的纯计算行为。
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def identity() -> ActorIdentity:
    return ActorIdentity(user_id="test_user", agent_id="test_agent", session_id="test_session")


@pytest.fixture
def identity_scope():
    """生成请求必须显式携带创建 Workspace，而不能从对话 ActorIdentity 猜测。"""
    return make_memory_identity_scope(user_id="test_user", agent_id="test_agent")


@pytest.fixture
def sample_messages(identity) -> list:
    return [
        StreamMessage(
            message_type=StreamMessageType.USER,
            content="帮我把 API 端口改成 9090",
            identity=identity,
        ),
        StreamMessage(
            message_type=StreamMessageType.ASSISTANT,
            content="好的，已修改端口配置",
            identity=identity,
        ),
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
        meta=make_memory_metadata(
            user_id=identity.user_id,
            source_agent_id=identity.agent_id,
            session_id=None,  # session_id 仅为兼容字段，不参与当前身份作用域传播
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


# ========== Test 4：Mode C 合并提示词 ==========


class TestModeCMergePrompt:
    """验证 Generation Engine Mode C 路径调用 extractor.merge()"""

    @pytest.mark.asyncio
    async def test_mode_c_calls_merge_not_extract(
        self, sample_context, existing_memory, identity_scope
    ):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="端口改为 9090",
            changelog="更新端口",
        )
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage,
            extractor=mock_extractor,
            deduplicator=MagicMock(),
        )

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
        result = await engine.process(request=request, identity_scope=identity_scope)

        # merge() 被调用，extract() 不被调用（Mode C 路由契约）
        mock_extractor.merge.assert_called_once()
        mock_extractor.extract.assert_not_called()

        # merge 参数携带 update 语义
        call_args = mock_extractor.merge.call_args
        metadata = call_args[1]["metadata"]
        assert metadata["mode"] == "update"
        assert metadata["instruction"] == "把端口改成 9090"
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mode_c_returns_updated_memory(self, existing_memory, identity_scope):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = MergeResult(
            new_content="新内容",
            changelog="测试更新",
        )
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage,
            extractor=mock_extractor,
            deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="更新",
            base_uuid=str(existing_memory.id),
            base_alias="fact_api_port",
        )
        request = GenerationRequest(
            update_focus=uf,
            existing_memory=existing_memory,
        )
        result = await engine.process(request=request, identity_scope=identity_scope)

        assert len(result) == 1
        assert result[0].atom.payload.content == "新内容"
        assert result[0].atom.get_alias() == existing_memory.get_alias()
        assert str(result[0].atom.id) == str(existing_memory.id)


# ========== Test 5：Mode C 回退 ==========


class TestModeCFallback:
    """验证 LLM 合并失败时的 fallback 拼接"""

    @pytest.mark.asyncio
    async def test_fallback_when_merge_returns_none(self, existing_memory, identity_scope):
        mock_extractor = MagicMock()
        mock_extractor.merge.return_value = None  # LLM 失败
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage,
            extractor=mock_extractor,
            deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="追加新内容",
            content="新增的段落",
            base_uuid=str(existing_memory.id),
            base_alias="fact_api_port",
        )
        request = GenerationRequest(
            update_focus=uf,
            existing_memory=existing_memory,
        )
        result = await engine.process(request=request, identity_scope=identity_scope)

        # fallback 应该保底入库
        assert len(result) == 1
        # fallback 拼接: 旧内容 + 新内容
        assert "新增的段落" in result[0].atom.payload.content
        assert existing_memory.payload.content.split("\n")[0] in result[0].atom.payload.content

    def test_fallback_content_append(self, existing_memory):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="追加内容",
            content="新段落文本",
            base_uuid="uuid-123",
            base_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory, now=FIXED_NOW)

        assert isinstance(result, MergeResult)
        assert "新段落文本" in result.new_content
        assert existing_memory.payload.content in result.new_content
        # fallback 追加段落使用传入 now 的日期
        assert "## 更新 (2026-09-01)" in result.new_content
        assert "Fallback" in result.changelog

    def test_fallback_instruction_only(self, existing_memory):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        uf = UpdateFocus(
            instruction="删除过时信息",
            content=None,
            base_uuid="uuid-123",
            base_alias="alias",
        )
        result = engine._build_update_fallback(uf, existing_memory, now=FIXED_NOW)

        # 无 content 时保留旧内容不变
        assert result.new_content == existing_memory.payload.content
        assert "Fallback" in result.changelog
        assert "删除过时信息" in result.changelog

    @pytest.mark.asyncio
    async def test_mode_c_no_existing_memory_returns_empty(self, identity_scope):
        """existing_memory 未注入时应返回空列表"""
        mock_extractor = MagicMock()
        mock_storage = _mock_mid_term()

        engine = MemoryGenerationEngine(
            mid_term=mock_storage,
            extractor=mock_extractor,
            deduplicator=MagicMock(),
        )

        uf = UpdateFocus(
            instruction="test",
            base_uuid="uuid-123",
            base_alias="alias",
        )
        # 不注入 existing_memory (默认 None)

        request = GenerationRequest(update_focus=uf)
        result = await engine.process(request=request, identity_scope=identity_scope)

        assert result == []
        mock_extractor.merge.assert_not_called()


# ========== Test 6：_apply_update 纯计算边界 ==========


class TestApplyUpdate:
    """验证引擎纯计算边界（内容合并 + before 快照）。

    MVL-2 起 ``meta.version``/``updated_at``/``decay_anchor_at``/
    ``confidence_score`` 由 Familiar 在提交边界分配，引擎层测试改为断言
    引擎"不写"这些字段。
    """

    def test_version_not_touched_by_engine(self, existing_memory, merge_result):
        mock_storage = _mock_mid_term()
        engine = MemoryGenerationEngine(
            mid_term=mock_storage,
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        updated_at_before = existing_memory.meta.updated_at
        decay_anchor_before = existing_memory.meta.lifecycle.decay_anchor_at

        result = engine._apply_update(
            existing_memory,
            merge_result,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        assert len(result) == 1
        # 引擎不推进版本与内容时间（version += 1 / updated_at 由 Familiar 提交边界负责）
        assert result[0].atom.meta.version == 1  # fixture 起点 version=1
        assert result[0].atom.meta.updated_at == updated_at_before
        assert result[0].atom.meta.lifecycle.decay_anchor_at == decay_anchor_before

    def test_content_updated(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        result = engine._apply_update(
            existing_memory,
            merge_result,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        assert result[0].atom.payload.content == merge_result.new_content

    def test_outcome_records_changelog_and_before_snapshot(self, existing_memory, merge_result):
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        result = engine._apply_update(
            existing_memory,
            merge_result,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        # 修改前完整原子以深拷贝形式随 outcome 返回，changelog 记录在 outcome 上
        snapshot = result[0].memory_before_snapshot
        assert snapshot is not existing_memory
        assert snapshot.payload.content == "API 服务运行在端口 8080，使用 HTTP 协议。"
        assert snapshot.meta.version == 1
        assert result[0].changelog == merge_result.changelog

    def test_confidence_untouched_by_engine(self, existing_memory, merge_result):
        """置信度重置 1.0 移到 Familiar 提交边界；引擎保持原值"""
        existing_memory.meta.lifecycle.confidence_score = 0.5
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )
        result = engine._apply_update(
            existing_memory,
            merge_result,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        assert result[0].atom.meta.lifecycle.confidence_score == 0.5

    def test_multiple_updates_snapshot_each_round(self, existing_memory):
        """多轮修订：每轮 before 快照捕获当次修改前的完整原子，版本仍归 Familiar"""
        engine = MemoryGenerationEngine(
            mid_term=_mock_mid_term(),
            extractor=MagicMock(),
            deduplicator=MagicMock(),
        )

        # 第一次更新
        r1 = MergeResult(new_content="v2 content", changelog="first update")
        r1_outcome = engine._apply_update(
            existing_memory,
            r1,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        # 第二次更新
        r2 = MergeResult(new_content="v3 content", changelog="second update")
        r2_outcome = engine._apply_update(
            existing_memory,
            r2,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        assert existing_memory.payload.content == "v3 content"
        # 引擎不推进版本：多轮修订后 version 仍由提交边界分配
        assert existing_memory.meta.version == 1
        # 每轮 before 快照捕获当次修改前的完整原子（v2 基准是 v1，v3 基准是 v2 内容）
        assert r1_outcome[0].memory_before_snapshot.payload.content == (
            "API 服务运行在端口 8080，使用 HTTP 协议。"
        )
        assert r1_outcome[0].memory_before_snapshot.meta.version == 1
        assert r2_outcome[0].memory_before_snapshot.payload.content == "v2 content"
        assert r2_outcome[0].memory_before_snapshot.meta.version == 1


# ========== Test 11：Active Flush 原因已移除 ==========


class TestTriggerReasonActiveGenerationRemoved:
    """主动更新生成已脱离感知层，不再保留 MTP flush reason"""

    def test_mtp_update_removed(self):
        assert "MTP_UPDATE" not in TriggerReason.__members__
