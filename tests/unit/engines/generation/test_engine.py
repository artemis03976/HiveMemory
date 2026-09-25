"""
MemoryGenerationEngine 单元测试

测试覆盖:
- Mode 路由: A (被动) / B (WRITE) / C (UPDATE)
- Mode A: LLM 提取成功 / 无价值 / 返回 None
- Mode B: 正常 WRITE / LLM 失败 fallback
- Mode C: 正常 UPDATE / existing_memory=None / LLM 合并失败 fallback
- 查重分支: TOUCH / UPDATE / CREATE / DISCARD
- 别名构建: 有 suffix / 从 title 派生 / 未知类型
- 纯计算边界: 引擎不改 version/updated_at/confidence，CREATE 用传入 now 打戳
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

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
    WriteFocus,
)
from hivememory.core.models.artifact import ArtifactRef, ArtifactType
from hivememory.core.models.memory import MEMORY_SUMMARY_MAX_LENGTH
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationContext,
    GenerationTurn,
    MergeResult,
    provenance_from_actor,
    system_settlement_provenance,
)
from hivememory.engines.generation.models import (
    GenerationRequest as GenerationRequestModel,
)
from tests.helpers.memory import make_memory_identity_scope, make_memory_metadata

GenerationRequest = GenerationRequestModel

# 提交边界固定时点：验证引擎对传入 now 的纯计算行为。
FIXED_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def _make_identity() -> ActorIdentity:
    return ActorIdentity(user_id="u1", agent_id="a1")


def _make_messages(n=2) -> list:
    identity = _make_identity()
    msgs = []
    for i in range(n):
        msg_type = StreamMessageType.USER if i % 2 == 0 else StreamMessageType.ASSISTANT
        msgs.append(StreamMessage(message_type=msg_type, content=f"msg_{i}", identity=identity))
    return msgs


def _make_context_from_messages(messages: list[StreamMessage]) -> GenerationContext:
    turns = []
    for i in range(0, len(messages), 2):
        user_msg = messages[i] if i < len(messages) else None
        assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
        turns.append(
            GenerationTurn(
                user_query=user_msg.content if user_msg else "",
                assistant_final_text=assistant_msg.content if assistant_msg else "",
                identity=(
                    assistant_msg.identity
                    if assistant_msg and assistant_msg.identity
                    else (user_msg.identity if user_msg and user_msg.identity else _make_identity())
                ),
            )
        )
    return GenerationContext(turns=turns)


def _make_context_with_agents(agent_ids: list[str]) -> GenerationContext:
    """构造每个轮次携带指定 Agent 身份的生成上下文。"""
    return GenerationContext(
        turns=[
            GenerationTurn(
                user_query=f"q_{i}",
                assistant_final_text=f"a_{i}",
                identity=ActorIdentity(user_id="u1", agent_id=agent_id),
            )
            for i, agent_id in enumerate(agent_ids)
        ]
    )


def _make_draft(
    has_value=True, title="测试记忆", alias_suffix="test_alias"
) -> ExtractedMemoryDraft:
    return ExtractedMemoryDraft(
        title=title,
        summary="这是一段足够长的测试摘要用于通过验证",
        tags=["t1"],
        memory_type="FACT",
        content="内容",
        confidence_score=0.9,
        has_value=has_value,
        alias_suffix=alias_suffix,
    )


def _make_memory(title="已有记忆") -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id="a1",
            user_id="u1",
            session_id="s1",
        ),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="旧内容"),
    )


class TestGenerationEngineRouting:
    """Mode 路由测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    @pytest.mark.asyncio
    async def test_empty_messages_no_focus_returns_empty(self):
        """空消息且无 focus 时早返回"""
        request = GenerationRequest()
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())
        assert result == []
        self.mock_extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_to_mode_a(self):
        """无 focus 时走 Mode A"""
        msgs = _make_messages()
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        self.mock_extractor.extract.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_routes_to_mode_b(self):
        """有 write_focus 时走 Mode B"""
        focus = WriteFocus(content="保存这段代码")
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=GenerationContext(), write_focus=focus)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        call_kwargs = self.mock_extractor.extract.call_args
        assert call_kwargs[1]["metadata"]["mode"] == "write"
        assert len(result) == 1
        assert result[0].atom.index.title == "测试记忆"

    @pytest.mark.asyncio
    async def test_routes_to_mode_c(self):
        """有 update_focus 时走 Mode C"""
        existing = _make_memory()
        uf = UpdateFocus(
            instruction="添加错误处理",
            base_uuid=str(existing.id),
            base_alias="fact_test",
        )
        merge_result = MergeResult(new_content="新内容", changelog="添加了错误处理")
        self.mock_extractor.merge.return_value = merge_result
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(
            context=GenerationContext(),
            update_focus=uf,
            existing_memory=existing,
        )
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        self.mock_extractor.merge.assert_called_once()
        assert len(result) == 1


class TestGenerationEngineModeA:
    """Mode A (被动观察) 测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    @pytest.mark.asyncio
    async def test_mode_a_extract_success(self):
        """正常提取流程"""
        msgs = _make_messages()
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert len(result) == 1
        assert result[0].atom.index.title == "测试记忆"

    @pytest.mark.asyncio
    async def test_mode_a_create_records_system_source_with_contributors(self):
        """被动结算没有具体 Agent 作为操作来源主体：来源为保留 system，
        单 Agent 内容的贡献者集合只包含实际参与内容的 Agent。"""
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        request = GenerationRequest(context=_make_context_with_agents(["a1"]))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "system"
        assert atom.meta.provenance.source_team_id is None
        assert atom.meta.provenance.contributing_agent_ids == ("a1",)

    @pytest.mark.asyncio
    async def test_mode_a_contributors_dedup_keep_order_and_exclude_system(self):
        """多 Agent 贡献按首次出现顺序去重，system 不是内容贡献者。"""
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        request = GenerationRequest(context=_make_context_with_agents(["b2", "a1", "b2", "system"]))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert result[0].atom.meta.provenance.contributing_agent_ids == ("b2", "a1")

    @pytest.mark.asyncio
    async def test_mode_a_dedup_update_merges_settle_contributors(self):
        """SETTLE 撞上已有记忆触发演化时，本轮结算的贡献者并入已有集合并进入
        Memory（Version Artifact 从 meta 拷贝）；来源字段按约定保留不改写。"""
        existing = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id="creator",
                user_id="u1",
                contributing_agent_ids=("creator",),
            ),
            index=IndexLayer(
                title="已有记忆",
                summary="这是一段足够长的测试摘要用于通过验证",
                tags=["t"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="旧内容"),
        )
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.UPDATE, existing)

        request = GenerationRequest(context=_make_context_with_agents(["b2", "a1"]))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        atom = result[0].atom
        assert atom is existing
        assert atom.meta.provenance.source_agent_id == "creator"
        assert atom.meta.provenance.contributing_agent_ids == ("creator", "b2", "a1")

    @pytest.mark.asyncio
    async def test_mode_a_extract_no_value(self):
        """LLM 判断无价值返回空"""
        msgs = _make_messages()
        draft = _make_draft(has_value=False)
        self.mock_extractor.extract.return_value = draft

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert result == []
        self.mock_deduplicator.check_duplicate.assert_not_called()

    @pytest.mark.asyncio
    async def test_mode_a_extract_returns_none(self):
        """LLM 返回 None 时返回空"""
        msgs = _make_messages()
        self.mock_extractor.extract.return_value = None

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert result == []

    @pytest.mark.asyncio
    async def test_mode_a_empty_messages(self):
        """Mode A 空消息列表返回空"""
        request = GenerationRequest()
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())
        assert result == []


class TestGenerationEngineModeB:
    """Mode B (WRITE 主动响应) 测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    @pytest.mark.asyncio
    async def test_mode_b_extract_success(self):
        """正常 WRITE 流程"""
        focus = WriteFocus(content="重要代码片段", reason="保存备用")
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=GenerationContext(), write_focus=focus)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_mode_b_create_seeds_actor_before_context_contributors(self):
        """WRITE 主动创建以提交操作的 actor 为来源；贡献者先记录发起 Agent，
        再合并上下文轮次贡献者。"""
        focus = WriteFocus(content="主动写入的内容", reason="保存")
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        request = GenerationRequest(
            context=_make_context_with_agents(["b2"]),
            write_focus=focus,
        )
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "a1"
        assert atom.meta.provenance.source_team_id is None
        assert atom.meta.provenance.contributing_agent_ids == ("a1", "b2")

    @pytest.mark.asyncio
    async def test_mode_b_create_without_context_records_actor_as_contributor(self):
        """无背景上下文的主动 WRITE：发起 Agent 本身仍作为内容贡献者记录。"""
        focus = WriteFocus(content="无上下文的主动写入", reason="保存")
        draft = _make_draft()
        self.mock_extractor.extract.return_value = draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        request = GenerationRequest(context=GenerationContext(), write_focus=focus)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "a1"
        assert atom.meta.provenance.contributing_agent_ids == ("a1",)

    @pytest.mark.asyncio
    async def test_mode_b_fallback_on_extract_failure(self):
        """LLM 提取失败时启用 fallback"""
        focus = WriteFocus(content="重要内容不能丢", reason="保存")
        self.mock_extractor.extract.return_value = None
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=GenerationContext(), write_focus=focus)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        # fallback 应保证内容不丢失
        assert len(result) == 1
        assert result[0].atom.payload.content == "重要内容不能丢"

    def test_build_fallback_draft(self):
        """fallback 草稿构建逻辑"""
        focus = WriteFocus(content="内容", title="标题", reason="原因")
        draft = self.engine._build_fallback_draft(focus)

        assert draft.title == "标题"
        # 摘要直接取 reason，不再为凑足长度拼接正文片段。
        assert draft.summary == "原因"
        assert draft.content == "内容"
        assert draft.has_value is True
        assert draft.confidence_score == 1.0
        assert "mtp_write" in draft.tags

    def test_build_fallback_draft_no_title(self):
        """fallback 无 title 时从 content 截取"""
        focus = WriteFocus(content="这是一段很长的内容用于测试")
        draft = self.engine._build_fallback_draft(focus)

        assert draft.title == "这是一段很长的内容用于测试"


class TestGenerationEngineModeC:
    """Mode C (UPDATE 合并更新) 测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    def _make_update_request(self, existing=None, instruction="更新内容", content=None):
        if existing is None:
            existing = _make_memory()
        uf = UpdateFocus(
            instruction=instruction,
            content=content,
            base_uuid=str(existing.id),
            base_alias="fact_test",
        )
        return GenerationRequest(
            context=GenerationContext(),
            update_focus=uf,
            existing_memory=existing,
        )

    @pytest.mark.asyncio
    async def test_mode_c_merge_success(self):
        """正常 UPDATE 合并流程"""
        merge_result = MergeResult(new_content="合并后内容", changelog="更新了内容")
        self.mock_extractor.merge.return_value = merge_result
        self.mock_storage.upsert = AsyncMock()

        request = self._make_update_request()
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert len(result) == 1
        assert result[0].atom.payload.content == "合并后内容"

    @pytest.mark.asyncio
    async def test_mode_c_update_merges_actor_and_context_contributors(self):
        """主动 UPDATE 保留已有来源字段，并把发起 Agent 与上下文贡献者并入集合。"""
        existing = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id="creator",
                user_id="u1",
                contributing_agent_ids=("creator",),
            ),
            index=IndexLayer(
                title="已有记忆",
                summary="这是一段足够长的测试摘要用于通过验证",
                tags=["t"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(content="旧内容"),
        )
        merge_result = MergeResult(new_content="合并后内容", changelog="更新了内容")
        self.mock_extractor.merge.return_value = merge_result
        self.mock_storage.upsert = AsyncMock()

        uf = UpdateFocus(
            instruction="更新内容",
            base_uuid=str(existing.id),
            base_alias="fact_test",
        )
        request = GenerationRequest(
            context=_make_context_with_agents(["b2"]),
            update_focus=uf,
            existing_memory=existing,
        )
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "creator"
        assert atom.meta.provenance.contributing_agent_ids == ("creator", "a1", "b2")

    @pytest.mark.asyncio
    async def test_mode_c_no_existing_memory(self):
        """existing_memory=None 时返回空"""
        uf = UpdateFocus(
            instruction="更新",
            base_uuid=str(uuid4()),
            base_alias="fact_test",
        )
        request = GenerationRequest(context=GenerationContext(), update_focus=uf)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert result == []
        self.mock_extractor.merge.assert_not_called()

    @pytest.mark.asyncio
    async def test_mode_c_fallback_on_merge_failure(self):
        """LLM 合并失败时启用 fallback"""
        self.mock_extractor.merge.return_value = None
        self.mock_storage.upsert = AsyncMock()

        request = self._make_update_request(content="追加内容")
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert len(result) == 1
        assert "追加内容" in result[0].atom.payload.content

    @pytest.mark.asyncio
    async def test_mode_c_fallback_no_content(self):
        """fallback 仅有 instruction 无 content 时保留旧内容"""
        existing = _make_memory()
        self.mock_extractor.merge.return_value = None
        self.mock_storage.upsert = AsyncMock()

        request = self._make_update_request(existing=existing, content=None)
        result = await self.engine.process(request, identity_scope=make_memory_identity_scope())

        assert len(result) == 1
        assert result[0].atom.payload.content == "旧内容"

    def test_apply_update_merges_content_and_captures_before_snapshot(self):
        """纯计算边界：内容合并 + before 快照；version/confidence 留给提交边界"""
        existing = _make_memory()
        existing.meta.lifecycle.confidence_score = 0.42
        version_before = existing.meta.version
        updated_at_before = existing.meta.updated_at
        merge_result = MergeResult(new_content="新版本", changelog="v2 更新")
        self.mock_storage.upsert = Mock()

        result = self.engine._apply_update(
            existing,
            merge_result,
            provenance=system_settlement_provenance(GenerationContext()),
        )

        assert len(result) == 1
        mem = result[0].atom
        # 引擎只合并内容，不分配版本与置信度（提交边界由 Familiar 负责）
        assert mem.payload.content == "新版本"
        assert mem.meta.version == version_before
        assert mem.meta.lifecycle.confidence_score == 0.42
        assert mem.meta.updated_at == updated_at_before
        # before 快照是深拷贝，保留修改前的完整原子
        snapshot = result[0].memory_before_snapshot
        assert snapshot is not existing
        assert snapshot.payload.content == "旧内容"
        assert snapshot.meta.version == version_before
        assert result[0].changelog == "v2 更新"

    def test_apply_update_degrades_to_touch_when_content_unchanged(self):
        """§3.2：合并结果与现有内容一致时不创建新版本——降级为 TOUCH 纯决策。"""
        existing = _make_memory()
        existing.payload.content = "旧内容"
        existing.meta.lifecycle.access_count = 3
        version_before = existing.meta.version
        updated_at_before = existing.meta.updated_at
        contributors_before = existing.meta.provenance.contributing_agent_ids
        # 裁定中携带新的贡献者：无内容变化时不得并入（贡献者只随受控内容提交合并）。
        provenance = MemoryProvenance(
            source_agent_id="a1",
            contributing_agent_ids=("new-contributor",),
        )
        merge_result = MergeResult(new_content="旧内容", changelog="Fallback (无变更)")

        result = self.engine._apply_update(
            existing,
            merge_result,
            provenance=provenance,
        )

        assert len(result) == 1
        outcome = result[0]
        assert outcome.duplicate_decision == DuplicateDecision.TOUCH
        # 零变更：内容、版本、内容时间、贡献者集合全部原样。
        assert outcome.atom is existing
        assert outcome.atom.payload.content == "旧内容"
        assert outcome.atom.meta.version == version_before
        assert outcome.atom.meta.updated_at == updated_at_before
        assert outcome.atom.meta.provenance.contributing_agent_ids == contributors_before
        assert outcome.memory_before_snapshot is None
        assert outcome.changelog is None


class TestGenerationEngineDedup:
    """查重分支测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    @pytest.mark.asyncio
    async def test_dedup_touch(self):
        """TOUCH 决策保持纯计算：不改任何字段，访问统计由 Familiar patch"""
        existing = _make_memory()
        existing.meta.lifecycle.access_count = 5
        draft = _make_draft()
        updated_at_before = existing.meta.updated_at
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.TOUCH, existing)

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope(),
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        self.mock_storage.upsert.assert_not_called()
        assert result[0].atom is existing
        assert result[0].duplicate_decision == DuplicateDecision.TOUCH
        # 引擎不改任何字段：访问计数与内容时间原样保留
        assert existing.meta.lifecycle.access_count == 5
        assert existing.meta.lifecycle.last_accessed_at is None
        assert existing.meta.updated_at == updated_at_before

    @pytest.mark.asyncio
    async def test_dedup_update(self):
        """UPDATE 决策覆盖当前 head，不持久化（持久化由 Familiar 负责）"""
        existing = _make_memory()
        existing_ref = ArtifactRef(
            artifact_id="ref1",
            artifact_type=ArtifactType.MEMORY_CREATION,
            workspace_identity=existing.workspace_identity,
        )
        existing.payload.artifacts.refs.append(existing_ref)
        old_version = existing.meta.version
        old_title = existing.index.title
        old_summary = existing.index.summary
        old_updated_at = existing.meta.updated_at
        draft = _make_draft(
            title="新版记忆",
            alias_suffix="new_alias",
        )
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.UPDATE, existing)

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope(),
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        self.mock_storage.upsert.assert_not_called()
        assert result[0].atom is existing
        assert result[0].duplicate_decision == DuplicateDecision.UPDATE
        assert existing.payload.artifacts.refs == [existing_ref]
        # 引擎不推进版本与内容时间（提交边界由 Familiar 负责）
        assert existing.meta.version == old_version
        assert existing.meta.updated_at == old_updated_at
        # 内容与检索层按草稿合并
        assert existing.payload.content == draft.content
        assert existing.index.title == draft.title
        assert existing.index.summary == draft.summary
        assert set(existing.index.tags) == {"t1", "t"}
        # before 快照保留修订前的 title/summary
        snapshot = result[0].memory_before_snapshot
        assert snapshot is not existing
        assert snapshot.index.title == old_title
        assert snapshot.index.summary == old_summary

    @pytest.mark.asyncio
    async def test_dedup_create(self):
        """CREATE 决策用传入 now 打创建时戳，不持久化（持久化由 Familiar 负责）"""
        draft = _make_draft()
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope(),
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        self.mock_storage.upsert.assert_not_called()
        assert len(result) == 1
        atom = result[0].atom
        assert atom.index.title == "测试记忆"
        # 创建时点 = 提交边界传入的 now（created/updated/decay 同值）
        assert atom.meta.created_at == FIXED_NOW
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW

    @pytest.mark.asyncio
    async def test_dedup_update_with_overlong_llm_summary_merges_truncated_index(self):
        """LLM 摘要越界时 dedup 合并写入截断值，而不是让生成失败或写入越界原子。"""
        existing = _make_memory()
        draft = _make_draft().model_dump()
        draft["summary"] = "摘" * 600
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.UPDATE, existing)

        result = await self.engine._dedup_and_resolve(
            ExtractedMemoryDraft(**draft),
            make_memory_identity_scope(),
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        assert result[0].duplicate_decision == DuplicateDecision.UPDATE
        assert result[0].atom.index.summary == "摘" * MEMORY_SUMMARY_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_dedup_discard(self):
        """DISCARD 决策返回空"""
        draft = _make_draft()
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISCARD, None)

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope(),
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        assert len(result) == 1
        assert result[0].atom is None
        assert result[0].duplicate_decision == DuplicateDecision.DISCARD
        self.mock_storage.upsert.assert_not_called()


class TestGenerationEngineAlias:
    """别名构建测试"""

    def test_build_alias_with_suffix(self):
        """有 alias_suffix 时使用 LLM 生成的后缀"""
        alias = MemoryGenerationEngine._build_alias("CODE_SNIPPET", "quicksort_impl", "快排实现")
        assert alias == "code_quicksort_impl"

    def test_build_alias_fallback_to_title(self):
        """无 suffix 时从 title 派生"""
        alias = MemoryGenerationEngine._build_alias("FACT", "", "Python Tips")
        assert alias == "fact_python_tips"

    def test_build_alias_unknown_type(self):
        """未知类型用 'mem' 前缀"""
        alias = MemoryGenerationEngine._build_alias("UNKNOWN_TYPE", "test", "标题")
        assert alias.startswith("mem_")

    def test_build_alias_empty_suffix_and_title(self):
        """suffix 和 title 都为空时返回 None"""
        alias = MemoryGenerationEngine._build_alias("FACT", "", "")
        assert alias is None

    def test_build_alias_cleans_special_chars(self):
        """清洗特殊字符"""
        alias = MemoryGenerationEngine._build_alias("FACT", "hello@world!!", "标题")
        assert alias == "fact_helloworld"

    def test_build_alias_truncates_long_suffix(self):
        """长 suffix 截断到 40 字符"""
        long_suffix = "a" * 100
        alias = MemoryGenerationEngine._build_alias("FACT", long_suffix, "标题")
        # 前缀 "fact_" + 40 字符
        assert len(alias) == 45


class TestGenerationEngineHelpers:
    """辅助方法测试"""

    def setup_method(self):
        self.engine = MemoryGenerationEngine(
            mid_term=Mock(),
            extractor=Mock(),
            deduplicator=Mock(),
        )

    def test_render_transcript(self):
        """统一 transcript 渲染入口"""
        transcript = self.engine._render_transcript(
            GenerationRequest(context=_make_context_from_messages(_make_messages(2)))
        )

        assert "[User]:" in transcript
        assert "[Assistant]:" in transcript

    def test_render_transcript_empty_context_placeholder(self):
        """空上下文返回统一占位文本"""
        transcript = self.engine._render_transcript(GenerationRequest())
        assert transcript == "(无背景对话)"

    def test_draft_to_memory(self):
        """草稿按 provenance 裁定来源字段，创建时点使用传入 now"""
        draft = _make_draft(title="测试标题")
        identity_scope = make_memory_identity_scope()
        provenance = provenance_from_actor(identity_scope, _make_context_with_agents(["a1"]))

        memory = self.engine._draft_to_memory(draft, identity_scope, provenance, now=FIXED_NOW)

        assert memory.index.title == "测试标题"
        assert memory.workspace_identity.owner_user_id == "u1"
        assert memory.meta.provenance.source_agent_id == "a1"
        assert memory.meta.provenance.contributing_agent_ids == ("a1",)
        assert memory.meta.lifecycle.confidence_score == 0.9
        # 创建时点 = 提交边界传入的 now（created/updated/decay 同值）
        assert memory.meta.created_at == FIXED_NOW
        assert memory.meta.updated_at == FIXED_NOW
        assert memory.meta.lifecycle.decay_anchor_at == FIXED_NOW

    def test_draft_to_memory_unknown_type(self):
        """未知记忆类型 fallback 到 FACT"""
        draft = _make_draft()
        draft.memory_type = "INVALID_TYPE"
        identity_scope = make_memory_identity_scope()

        memory = self.engine._draft_to_memory(
            draft,
            identity_scope,
            system_settlement_provenance(GenerationContext()),
            now=FIXED_NOW,
        )

        assert memory.index.memory_type == MemoryType.FACT
