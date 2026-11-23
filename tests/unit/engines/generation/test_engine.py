"""
MemoryGenerationEngine 单元测试

测试覆盖:
- Mode 路由: A (被动) / B (WRITE) / C (UPDATE)
- Mode A: 多条草稿逐条查重 / 无价值草稿丢弃 / 没有草稿 / 同批别名不冲突
- Mode B: 模型草稿新建且置信度为 1.0 / 不合格时带错误重试一次 / 仍不合格、提供商失败与
  没有模型时兜底
- Mode C: 编辑集修订 / existing_memory=None / 带错误重试一次 / 重试后仍不合格、提供商失败与
  没有模型时回退且不改索引 / 空编辑集降为 TOUCH
- 查重分支: IDENTICAL / SIMILAR / DISTINCT；命中本话题主动写入结算到的记忆时 Mode A 丢弃、
  Mode B 不提名合并
- 段引用: 标签换算为轮次标识加事件区间，不存在的标签丢弃；退回草稿、追加修订与触碰不带
- 结算方式: 一次性生成查重；馆长的草稿全部新建、带段引用、同批别名不冲突，修订成为 UPDATE；
  提取关闭时都不运行
- 合并: 以保留记忆为底的编辑集，类型沿用保留记忆，空编辑集仍是 UPDATE，拒绝向上抛出
- 纯计算边界: 引擎不改 version/updated_at/decay anchor/confidence，CREATE 用传入 now 打戳；
  修订与合并不修改输入
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from hivememory.config.patchouli import ExtractorConfig, MemoryGenerationConfig
from hivememory.core.errors import ModelProviderError
from hivememory.core.models import (
    ActorIdentity,
    CitedSegment,
    IndexLayer,
    MemoryAccessPolicy,
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
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.edits import EditSetError
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.extractor import (
    LLMMemoryExtractor,
    NoOpMemoryExtractor,
    create_extractor,
)
from hivememory.engines.generation.librarian import create_librarian
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationContext,
    GenerationSegment,
    GenerationTurn,
    MemoryEditSet,
    provenance_from_actor,
    system_settlement_provenance,
)
from hivememory.engines.generation.models import (
    GenerationRequest as GenerationRequestModel,
)
from hivememory.infrastructure.llm.base import LLMRequestParams
from tests.helpers.llm import ScriptedLLMService
from tests.helpers.memory import (
    make_memory_identity_scope,
    make_memory_metadata,
)

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
                turn_id=f"turn_{i}",
                user_message=user_msg.content if user_msg else "",
                material_entries=[f"[Assistant]: {assistant_msg.content}"] if assistant_msg else [],
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
                turn_id=f"turn_{i}",
                user_message=f"q_{i}",
                material_entries=[f"[Assistant]: a_{i}"],
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


def _append(text: str, *, segments: tuple[str, ...] = ()) -> MemoryEditSet:
    """在正文末尾追加一段的编辑集。"""
    return MemoryEditSet.model_validate(
        {
            "edits": [{"op": "append", "text": text}],
            "changelog": f"追加{text}",
            "source_segments": list(segments),
        }
    )


def _make_memory(title="已有记忆", alias: str | None = None) -> MemoryAtom:
    return MemoryAtom(
        meta=make_memory_metadata(
            source_agent_id="a1",
            user_id="u1",
        ),
        index=IndexLayer(
            title=title,
            summary="这是一段足够长的测试摘要用于通过验证",
            tags=["t"],
            memory_type=MemoryType.FACT,
            alias=alias,
        ),
        payload=PayloadLayer(content="旧内容"),
    )


class TestGenerationEngineRouting:
    """Mode 路由测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = AsyncMock()
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
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )
        assert result == []
        self.mock_extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_to_mode_a(self):
        """无 focus 时走 Mode A"""
        msgs = _make_messages()
        draft = _make_draft()
        self.mock_extractor.extract.return_value = [draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        self.mock_extractor.extract.assert_called_once()
        self.mock_extractor.extract_write.assert_not_called()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_routes_to_mode_b(self):
        """有 write_focus 时走 Mode B，结构化调用携带写入内容、理由与 agent 拟定的标题"""
        focus = WriteFocus(content="保存这段代码", reason="后续复用", title="排序函数的实现")
        self.mock_extractor.extract_write.return_value = _make_draft().model_dump_json()
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=GenerationContext(), write_focus=focus)
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        kwargs = self.mock_extractor.extract_write.call_args.kwargs
        assert (kwargs["content"], kwargs["reason"], kwargs["title"]) == (
            "保存这段代码",
            "后续复用",
            "排序函数的实现",
        )
        self.mock_extractor.extract.assert_not_called()
        assert len(result) == 1
        assert result[0].atom.index.title == "测试记忆"

    @pytest.mark.asyncio
    async def test_routes_to_mode_c(self):
        """有 update_focus 时走 Mode C：调用 revise 而非 extract，并携带修订指令"""
        existing = _make_memory()
        uf = UpdateFocus(
            base_version=1,
            instruction="添加错误处理",
            base_uuid=str(existing.id),
            base_alias="fact_test",
        )
        self.mock_extractor.revise.return_value = _append("错误处理").model_dump_json()
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(
            context=GenerationContext(),
            update_focus=uf,
            existing_memory=existing,
        )
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        self.mock_extractor.revise.assert_called_once()
        self.mock_extractor.extract.assert_not_called()
        assert self.mock_extractor.revise.call_args.kwargs["instruction"] == "添加错误处理"
        assert len(result) == 1


class TestGenerationEngineModeA:
    """Mode A (被动观察) 测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = AsyncMock()
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
        self.mock_extractor.extract.return_value = [draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)
        self.mock_storage.upsert = AsyncMock()

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert len(result) == 1
        assert result[0].atom.index.title == "测试记忆"

    @pytest.mark.asyncio
    async def test_mode_a_create_records_system_source_with_contributors(self):
        """被动结算没有具体 Agent 作为操作来源主体：来源为保留 system，
        单 Agent 内容的贡献者集合只包含实际参与内容的 Agent。"""
        draft = _make_draft()
        self.mock_extractor.extract.return_value = [draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)

        request = GenerationRequest(context=_make_context_with_agents(["a1"]))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "system"
        assert atom.meta.provenance.source_team_id is None
        assert atom.meta.provenance.contributing_agent_ids == ("a1",)

    @pytest.mark.asyncio
    async def test_mode_a_contributors_dedup_keep_order_and_exclude_system(self):
        """多 Agent 贡献按首次出现顺序去重，system 不是内容贡献者。"""
        draft = _make_draft()
        self.mock_extractor.extract.return_value = [draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)

        request = GenerationRequest(context=_make_context_with_agents(["b2", "a1", "b2", "system"]))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert result[0].atom.meta.provenance.contributing_agent_ids == ("b2", "a1")

    @pytest.mark.asyncio
    async def test_mode_a_dedup_merge_creates_with_settle_contributors(self):
        """SETTLE 命中相似记忆后仍以 system 为来源新建，本轮贡献者记录在新记忆中。"""
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
        self.mock_extractor.extract.return_value = [draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.SIMILAR, existing)

        request = GenerationRequest(context=_make_context_with_agents(["b2", "a1"]))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        atom = result[0].atom
        assert atom.id != existing.id
        assert atom.meta.provenance.source_agent_id == "system"
        assert atom.meta.provenance.contributing_agent_ids == ("b2", "a1")
        assert result[0].merge_target_id == str(existing.id)

    @pytest.mark.asyncio
    async def test_mode_a_resolves_each_draft_with_distinct_aliases(self):
        """一段话题的多条草稿按顺序逐条查重并各自产出结果；无价值的草稿不进入查重。

        同一批草稿尚未落库，后缀相同时第二条仍要得到消歧后的别名。
        """
        existing = _make_memory()
        self.mock_extractor.extract.return_value = [
            _make_draft(title="决定", alias_suffix="same_topic"),
            _make_draft(has_value=False, title="寒暄"),
            _make_draft(title="进展", alias_suffix="same_topic"),
        ]
        self.mock_deduplicator.check_duplicate.side_effect = [
            (DuplicateDecision.DISTINCT, None),
            (DuplicateDecision.SIMILAR, existing),
        ]

        request = GenerationRequest(context=_make_context_from_messages(_make_messages()))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert [(o.action, o.atom.index.title, o.atom.index.alias) for o in result] == [
            (GenerationAction.CREATE, "决定", "fact_same_topic"),
            (GenerationAction.CREATE, "进展", "fact_same_topic_2"),
        ]
        assert [o.merge_target_id for o in result] == [None, str(existing.id)]

    @pytest.mark.asyncio
    async def test_mode_a_drops_drafts_hitting_this_topics_written_memories(self):
        """命中本话题主动写入结算到的记忆时，相似或相同都丢弃草稿：不新建、不触碰、
        不提名合并；命中其他记忆的草稿照常新建并提名合并。"""
        written_similar = _make_memory(alias="fact_written_one")
        written_identical = _make_memory(alias="fact_written_two")
        other = _make_memory(alias="fact_other")
        self.mock_extractor.extract.return_value = [
            _make_draft(title="复述一", alias_suffix="restated_one"),
            _make_draft(title="复述二", alias_suffix="restated_two"),
            _make_draft(title="新发现", alias_suffix="new_finding"),
        ]
        self.mock_storage.search = AsyncMock(return_value=[{"memory": other, "score": 0.8}])
        self.mock_deduplicator.check_duplicate.side_effect = [
            (DuplicateDecision.SIMILAR, written_similar),
            (DuplicateDecision.IDENTICAL, written_identical),
            (DuplicateDecision.SIMILAR, other),
        ]

        request = GenerationRequest(
            context=_make_context_from_messages(_make_messages()),
            topic_write_memory_ids=frozenset({str(written_similar.id), str(written_identical.id)}),
        )
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert [(o.action, o.atom.index.title, o.merge_target_id) for o in result] == [
            (GenerationAction.CREATE, "新发现", str(other.id))
        ]

    @pytest.mark.asyncio
    async def test_mode_a_extract_no_value(self):
        """LLM 判断无价值返回空"""
        msgs = _make_messages()
        draft = _make_draft(has_value=False)
        self.mock_extractor.extract.return_value = [draft]

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert result == []
        self.mock_deduplicator.check_duplicate.assert_not_called()

    @pytest.mark.asyncio
    async def test_mode_a_without_drafts_returns_empty(self):
        """模型没有给出草稿时没有产出"""
        msgs = _make_messages()
        self.mock_extractor.extract.return_value = []

        request = GenerationRequest(context=_make_context_from_messages(msgs))
        result = await self.engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_mode_a_provider_failure_fails_instead_of_producing_nothing(self):
        """被动结算的提供商失败向上传播，由生成任务按失败收尾，不记录为没有产出。"""
        failure = ModelProviderError("provider down")
        self.mock_extractor.extract.side_effect = failure

        request = GenerationRequest(context=_make_context_from_messages(_make_messages()))
        with pytest.raises(ModelProviderError) as raised:
            await self.engine.process(
                request,
                belong_to=make_memory_identity_scope().workspace_identity,
                from_actor=make_memory_identity_scope().actor_identity,
            )

        assert raised.value is failure
        self.mock_deduplicator.check_duplicate.assert_not_called()


class TestGenerationEngineModeB:
    """Mode B (WRITE 主动响应)：模型写出草稿；不能用作草稿时带着错误重试一次，
    仍不能用、提供商失败或没有模型时以兜底草稿保住写入的内容。"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock(
            return_value=(DuplicateDecision.DISTINCT, None)
        )

    def _engine(self, *outputs, extractor=None):
        return MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=extractor
            or LLMMemoryExtractor(
                ExtractorConfig(), ScriptedLLMService(*outputs), LLMRequestParams(model="m")
            ),
            deduplicator=self.mock_deduplicator,
        )

    async def _write(self, engine, *, content="写入的内容", context=None):
        request = GenerationRequest(
            context=context or GenerationContext(),
            write_focus=WriteFocus(content=content, reason="保存"),
        )
        return await engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

    @pytest.mark.asyncio
    async def test_mode_b_creates_the_model_draft_with_full_confidence(self):
        """模型的草稿按原样新建；明确的保存意图强制入库，置信度为 1.0，不采用模型的判断。"""
        draft = _make_draft(has_value=False).model_copy(update={"confidence_score": 0.3})

        [outcome] = await self._write(self._engine(draft.model_dump_json()))

        atom = outcome.atom
        assert (outcome.action, atom.index.title, atom.index.tags) == (
            GenerationAction.CREATE,
            "测试记忆",
            ["t1"],
        )
        assert atom.meta.lifecycle.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_mode_b_create_seeds_actor_before_context_contributors(self):
        """WRITE 主动创建以提交操作的 actor 为来源；贡献者先记录发起 Agent，
        再合并上下文轮次贡献者。"""
        engine = self._engine(_make_draft().model_dump_json())

        result = await self._write(engine, context=_make_context_with_agents(["b2"]))

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "a1"
        assert atom.meta.provenance.source_team_id is None
        assert atom.meta.provenance.contributing_agent_ids == ("a1", "b2")

    @pytest.mark.asyncio
    async def test_mode_b_create_without_context_records_actor_as_contributor(self):
        """无背景上下文的主动 WRITE：发起 Agent 本身仍作为内容贡献者记录。"""
        result = await self._write(self._engine(_make_draft().model_dump_json()))

        atom = result[0].atom
        assert atom.meta.provenance.source_agent_id == "a1"
        assert atom.meta.provenance.contributing_agent_ids == ("a1",)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first_output", "reason_in_retry"),
        [
            ("这不是 JSON", "not a JSON object"),
            (json.dumps(_make_draft().model_dump(exclude={"content"})), "content"),
        ],
        ids=["unparsable", "missing-field"],
    )
    async def test_mode_b_retries_once_with_the_error_before_falling_back(
        self, first_output, reason_in_retry
    ):
        """第一次的输出解析不了或缺少字段：把它与被拒的原因交回模型再要一次，
        第二次合格就按模型的草稿新建，而不是兜底草稿。"""
        llm = ScriptedLLMService(first_output, _make_draft().model_dump_json())
        engine = self._engine(
            extractor=LLMMemoryExtractor(ExtractorConfig(), llm, LLMRequestParams(model="m"))
        )

        [outcome] = await self._write(engine)

        assert (outcome.atom.index.title, outcome.atom.index.tags) == ("测试记忆", ["t1"])
        assert reason_in_retry in llm.calls[-1].messages[-1]["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outputs", "noop"),
        [
            (["这不是 JSON", "仍然不是 JSON"], False),
            ([ModelProviderError("down"), ModelProviderError("still down")], False),
            ([], True),
        ],
        ids=["still-invalid", "provider-failure", "no-model"],
    )
    async def test_mode_b_falls_back_to_the_written_content(self, outputs, noop):
        """重试后仍不能用、提供商失败或没有模型时启用兜底草稿，明确的保存意图不丢失。"""
        engine = self._engine(*outputs, extractor=NoOpMemoryExtractor() if noop else None)

        [outcome] = await self._write(engine, content="重要内容不能丢")

        assert (outcome.atom.payload.content, outcome.atom.index.tags) == (
            "重要内容不能丢",
            ["mtp_write"],
        )

    @pytest.mark.asyncio
    async def test_mode_b_program_error_is_not_replaced_by_fallback(self):
        """程序错误不属于 LLM 失败，不能由 fallback 掩盖。"""
        failure = TypeError("broken prompt wiring")
        extractor = AsyncMock()
        extractor.extract_write.side_effect = failure

        with pytest.raises(TypeError) as raised:
            await self._write(self._engine(extractor=extractor))

        assert raised.value is failure

    def test_build_fallback_draft(self):
        """fallback 草稿构建逻辑"""
        focus = WriteFocus(content="内容", title="标题", reason="原因")
        draft = self._engine()._build_fallback_draft(focus)

        assert draft.title == "标题"
        # 摘要直接取 reason，不再为凑足长度拼接正文片段。
        assert draft.summary == "原因"
        assert draft.content == "内容"
        assert draft.has_value is True
        assert draft.confidence_score == 1.0
        assert draft.memory_type == "FACT"
        assert "mtp_write" in draft.tags

    def test_build_fallback_draft_no_title(self):
        """fallback 无 title 时取 content 前 50 个字符"""
        focus = WriteFocus(content="A very long content that should be truncated for title")
        draft = self._engine()._build_fallback_draft(focus)

        assert draft.title == "A very long content that should be truncated for t"


def _edit_set_json(*edits: dict, add_tags=(), segments=(), changelog="改了") -> str:
    """模型输出的一份编辑集。"""
    return json.dumps(
        {
            "edits": list(edits),
            "add_tags": list(add_tags),
            "remove_tags": [],
            "changelog": changelog,
            "source_segments": list(segments),
        },
        ensure_ascii=False,
    )


def _replace(field: str, old: str, new: str) -> dict:
    return {"op": "replace", "field": field, "old": old, "new": new}


class TestGenerationEngineModeC:
    """Mode C (UPDATE 修订)：模型给出编辑集；不合格时带着错误重试一次，
    仍不合格、提供商失败或没有模型时回退，回退只改正文。"""

    def _engine(self, *outputs, extractor=None):
        return MemoryGenerationEngine(
            mid_term=Mock(),
            extractor=extractor
            or LLMMemoryExtractor(
                ExtractorConfig(), ScriptedLLMService(*outputs), LLMRequestParams(model="m")
            ),
            deduplicator=Mock(),
        )

    async def _update(self, engine, existing, *, content=None, context=None):
        request = GenerationRequest(
            context=context or GenerationContext(),
            update_focus=UpdateFocus(
                base_version=existing.meta.version,
                instruction="更新内容",
                content=content,
                base_uuid=str(existing.id),
                base_alias="fact_test",
            ),
            existing_memory=existing,
        )
        return await engine.process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
            now=FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_mode_c_applies_the_edit_set_to_the_existing_memory(self):
        """编辑集改正文、标题与标签；身份、类型、版本、内容时间与置信度不变，输入不被修改。"""
        existing = _make_memory(alias="fact_test")
        existing.meta.lifecycle.confidence_score = 0.42
        before = existing.model_copy(deep=True)
        engine = self._engine(
            _edit_set_json(
                _replace("content", "旧内容", "新内容"),
                _replace("title", "已有记忆", "已修订的记忆"),
                add_tags=["新"],
                changelog="改为新内容",
            )
        )

        [outcome] = await self._update(engine, existing)

        atom = outcome.atom
        assert outcome.action == GenerationAction.UPDATE
        assert (atom.index.title, atom.index.tags, atom.payload.content) == (
            "已修订的记忆",
            ["t", "新"],
            "新内容",
        )
        assert (atom.id, atom.index.alias, atom.index.memory_type) == (
            before.id,
            "fact_test",
            before.index.memory_type,
        )
        assert (
            atom.meta.version,
            atom.meta.updated_at,
            atom.meta.lifecycle.confidence_score,
        ) == (before.meta.version, before.meta.updated_at, 0.42)
        assert (outcome.memory_before_snapshot, outcome.changelog) == (before, "改为新内容")
        assert existing == before

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
        engine = self._engine(_edit_set_json(_replace("content", "旧内容", "新内容")))

        [outcome] = await self._update(engine, existing, context=_make_context_with_agents(["b2"]))

        atom = outcome.atom
        assert atom.meta.provenance.source_agent_id == "creator"
        assert atom.meta.provenance.contributing_agent_ids == ("creator", "a1", "b2")

    @pytest.mark.asyncio
    async def test_mode_c_no_existing_memory(self):
        """existing_memory=None 时返回空，不调用模型（替身没有预置输出，调用即失败）。"""
        request = GenerationRequest(
            context=GenerationContext(),
            update_focus=UpdateFocus(
                base_version=1, instruction="更新", base_uuid=str(uuid4()), base_alias="fact_test"
            ),
        )
        result = await self._engine().process(
            request,
            belong_to=make_memory_identity_scope().workspace_identity,
            from_actor=make_memory_identity_scope().actor_identity,
        )

        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first_output", "reason_in_retry"),
        [
            (_edit_set_json(_replace("content", "并不存在的原文", "x")), "edit 1"),
            ("这不是 JSON", "not a JSON object"),
        ],
        ids=["not-applicable", "unparsable"],
    )
    async def test_mode_c_retries_once_with_the_error_before_falling_back(
        self, first_output, reason_in_retry
    ):
        """第一次的输出不能应用或无法解析：把它与被拒的原因交回模型再要一次，
        第二次合格就照常修订。"""
        llm = ScriptedLLMService(
            first_output, _edit_set_json(_replace("content", "旧内容", "改对了"))
        )
        engine = self._engine(
            extractor=LLMMemoryExtractor(ExtractorConfig(), llm, LLMRequestParams(model="m")),
        )

        [outcome] = await self._update(engine, _make_memory(), content="追加内容")

        assert (outcome.action, outcome.atom.payload.content) == (
            GenerationAction.UPDATE,
            "改对了",
        )
        assert reason_in_retry in llm.calls[-1].messages[-1]["content"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outputs", "noop"),
        [
            ([_edit_set_json(_replace("content", "并不存在的原文", "x"))] * 2, False),
            ([ModelProviderError("down"), ModelProviderError("still down")], False),
            ([], True),
        ],
        ids=["still-invalid", "provider-failure", "no-model"],
    )
    async def test_mode_c_falls_back_to_appending_without_touching_the_index(self, outputs, noop):
        """重试后仍不合格、提供商失败或没有模型时，以提交时点日期把新内容追加到正文；
        标题、摘要与标签不变，内容不因模型没能写成编辑集而丢失。"""
        existing = _make_memory()
        engine = self._engine(*outputs, extractor=NoOpMemoryExtractor() if noop else None)

        [outcome] = await self._update(engine, existing, content="追加内容")

        assert outcome.atom.payload.content == "旧内容\n\n## 更新 (2026-09-01)\n追加内容"
        assert outcome.atom.index == existing.index
        assert outcome.changelog.startswith("Fallback")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("output", "content"),
        [(_edit_set_json(changelog="不需要改"), "追加内容"), (None, None)],
        ids=["empty-edit-set", "fallback-without-content"],
    )
    async def test_mode_c_without_a_change_degrades_to_touch(self, output, content):
        """空编辑集表示不需要改；回退时只有指令没有内容也不改。两者都降为 TOUCH，不建版本。"""
        existing = _make_memory()
        engine = (
            self._engine(output)
            if output is not None
            else self._engine(extractor=NoOpMemoryExtractor())
        )

        [outcome] = await self._update(engine, existing, content=content)

        assert (outcome.action, outcome.atom, outcome.memory_before_snapshot) == (
            GenerationAction.TOUCH,
            existing,
            None,
        )


class TestGenerationEngineDedup:
    """查重分支测试"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        self.mock_storage.upsert = AsyncMock()
        self.mock_extractor = AsyncMock()
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.check_duplicate = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )

    @pytest.mark.asyncio
    async def test_dedup_touch(self):
        """内容一致时返回 TOUCH 且保持纯计算：不改任何字段，访问统计由 Familiar patch"""
        existing = _make_memory()
        existing.meta.lifecycle.access_count = 5
        draft = _make_draft()
        updated_at_before = existing.meta.updated_at
        self.mock_deduplicator.check_duplicate.return_value = (
            DuplicateDecision.IDENTICAL,
            existing,
        )

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope().workspace_identity,
            system_settlement_provenance(GenerationContext()),
            from_actor=make_memory_identity_scope().actor_identity,
            now=FIXED_NOW,
            cited_segments=[],
        )

        self.mock_storage.upsert.assert_not_called()
        assert result[0].atom is existing
        assert result[0].action == GenerationAction.TOUCH
        # 引擎不改任何字段：访问计数与内容时间原样保留
        assert existing.meta.lifecycle.access_count == 5
        assert existing.meta.lifecycle.last_accessed_at is None
        assert existing.meta.updated_at == updated_at_before

    @pytest.mark.asyncio
    @pytest.mark.parametrize("restriction", [None, "policy", "topic_write"])
    async def test_dedup_merge_creates_before_nominating(self, restriction):
        """查重先创建独立记忆；只有策略相同，且命中的不是本话题主动写入结算到的记忆时才提名。"""
        existing = _make_memory(alias="fact_existing")
        existing.payload.artifacts.refs.append(
            ArtifactRef(
                artifact_id="ref1",
                artifact_type=ArtifactType.MEMORY_CREATION,
                workspace_identity=existing.workspace_identity,
            )
        )
        draft = _make_draft(title="新版记忆", alias_suffix="new_alias")
        if restriction == "policy":
            existing.meta.access_policy = MemoryAccessPolicy(
                visibility="PRIVATE", target_agent_id="a1"
            )
        topic_writes = (
            frozenset({str(existing.id)}) if restriction == "topic_write" else frozenset()
        )
        before = existing.model_copy(deep=True)
        self.mock_extractor.extract_write.return_value = draft.model_dump_json()
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.SIMILAR, existing)
        identity = make_memory_identity_scope()

        result = await self.engine.process(
            GenerationRequest(
                write_focus=WriteFocus(content=draft.content),
                topic_write_memory_ids=topic_writes,
            ),
            belong_to=identity.workspace_identity,
            from_actor=identity.actor_identity,
            now=FIXED_NOW,
        )

        outcome = result[0]
        atom = outcome.atom
        assert existing == before
        assert outcome.action == GenerationAction.CREATE
        assert outcome.merge_target_id == (str(existing.id) if restriction is None else None)
        assert outcome.memory_before_snapshot is None
        assert atom.id != existing.id
        assert atom.get_alias() != existing.get_alias()
        assert atom.payload.content == draft.content
        assert atom.index.title == draft.title
        assert atom.index.summary == draft.summary
        assert atom.index.tags == draft.tags
        assert atom.payload.artifacts.refs == []
        assert atom.meta.version == 1
        assert atom.meta.created_at == atom.meta.updated_at == FIXED_NOW
        assert atom.meta.access_policy == MemoryAccessPolicy.public()

    @pytest.mark.asyncio
    async def test_dedup_create(self):
        """低分候选仍为普通 CREATE，使用传入 now 打创建时戳且不提名合并。"""
        draft = _make_draft()
        self.mock_deduplicator.check_duplicate.return_value = (
            DuplicateDecision.DISTINCT,
            _make_memory(),
        )

        result = await self.engine._dedup_and_resolve(
            draft,
            make_memory_identity_scope().workspace_identity,
            system_settlement_provenance(GenerationContext()),
            from_actor=make_memory_identity_scope().actor_identity,
            now=FIXED_NOW,
            cited_segments=[],
        )

        self.mock_storage.upsert.assert_not_called()
        assert len(result) == 1
        assert result[0].merge_target_id is None
        atom = result[0].atom
        assert atom.index.title == "测试记忆"
        # 创建时点 = 提交边界传入的 now（created/updated/decay 同值）
        assert atom.meta.created_at == FIXED_NOW
        assert atom.meta.updated_at == FIXED_NOW
        assert atom.meta.lifecycle.decay_anchor_at == FIXED_NOW

    @pytest.mark.asyncio
    async def test_dedup_merge_with_overlong_llm_summary_creates_truncated_index(self):
        """LLM 摘要越界时新建记忆使用截断值，仍可提名合并。"""
        existing = _make_memory()
        draft = _make_draft().model_dump()
        draft["summary"] = "摘" * 600
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.SIMILAR, existing)

        result = await self.engine._dedup_and_resolve(
            ExtractedMemoryDraft(**draft),
            make_memory_identity_scope().workspace_identity,
            system_settlement_provenance(GenerationContext()),
            from_actor=make_memory_identity_scope().actor_identity,
            now=FIXED_NOW,
            cited_segments=[],
        )

        assert result[0].action == GenerationAction.CREATE
        assert result[0].atom.index.summary == "摘" * MEMORY_SUMMARY_MAX_LENGTH


def _segmented_context() -> GenerationContext:
    """两轮带段的材料：第 1 轮一段，第 2 轮两段，标签依次为 1.1、2.1、2.2。"""
    return GenerationContext(
        turns=[
            GenerationTurn(
                turn_id="turn-a",
                user_message="q1",
                material_entries=["[Assistant]: a1"],
                segments=[GenerationSegment(first_sequence=0, last_sequence=1, entry_count=1)],
                identity=_make_identity(),
            ),
            GenerationTurn(
                turn_id="turn-b",
                user_message="q2",
                material_entries=["[Assistant]: b1", "[Assistant]: b2"],
                segments=[
                    GenerationSegment(first_sequence=0, last_sequence=2, entry_count=1),
                    GenerationSegment(first_sequence=3, last_sequence=7, entry_count=1),
                ],
                identity=_make_identity(),
            ),
        ]
    )


class TestCitedSegments:
    """模型引用的段换算为轮次标识加事件区间，只随新建与修订交给提交。"""

    def setup_method(self):
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(return_value=[])
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        self.mock_extractor = AsyncMock()
        self.mock_deduplicator = Mock()
        self.engine = MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator,
        )
        self.identity = make_memory_identity_scope()

    async def _process(self, request: GenerationRequest):
        return await self.engine.process(
            request,
            belong_to=self.identity.workspace_identity,
            from_actor=self.identity.actor_identity,
            now=FIXED_NOW,
        )

    @pytest.mark.asyncio
    async def test_passive_drafts_cite_segments_and_unknown_labels_are_dropped(self):
        """被动草稿的标签按引用顺序去重后换算为区间，材料中不存在的标签丢弃；触碰不带引用。"""
        self.mock_extractor.extract.return_value = [
            _make_draft(title="决定").model_copy(
                update={"source_segments": ["2.2", "9.1", "1.1", "2.2"]}
            ),
            _make_draft(title="重复").model_copy(update={"source_segments": ["2.1"]}),
        ]
        self.mock_deduplicator.check_duplicate.side_effect = [
            (DuplicateDecision.DISTINCT, None),
            (DuplicateDecision.IDENTICAL, _make_memory()),
        ]

        created, touched = await self._process(GenerationRequest(context=_segmented_context()))

        assert created.cited_segments == [
            CitedSegment(turn_id="turn-b", first_sequence=3, last_sequence=7),
            CitedSegment(turn_id="turn-a", first_sequence=0, last_sequence=1),
        ]
        assert (touched.action, touched.cited_segments) == (GenerationAction.TOUCH, [])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_draft", "expected"),
        [
            (
                _make_draft().model_copy(update={"source_segments": ["2.1"]}).model_dump_json(),
                [CitedSegment(turn_id="turn-b", first_sequence=0, last_sequence=2)],
            ),
            (ModelProviderError("provider down"), []),
        ],
        ids=["model-draft", "fallback-draft"],
    )
    async def test_write_cites_only_with_a_model_draft(self, model_draft, expected):
        """主动写入的模型草稿带引用；提供商失败时退回的草稿没有模型引用的段。"""
        self.mock_extractor.extract_write.side_effect = [model_draft]
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISTINCT, None)

        [outcome] = await self._process(
            GenerationRequest(context=_segmented_context(), write_focus=WriteFocus(content="内容"))
        )

        assert (outcome.action, outcome.cited_segments) == (GenerationAction.CREATE, expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("revision", "expected"),
        [
            (
                _append("新内容", segments=("1.1",)).model_dump_json(),
                [CitedSegment(turn_id="turn-a", first_sequence=0, last_sequence=1)],
            ),
            (None, []),
        ],
        ids=["model-revision", "appended-fallback"],
    )
    async def test_revision_cites_only_with_a_model_revision(self, revision, expected):
        """修订带编辑集引用的段；没有模型时追加的修订不带引用。"""
        existing = _make_memory()
        self.mock_extractor.revise.return_value = revision
        request = GenerationRequest(
            context=_segmented_context(),
            update_focus=UpdateFocus(
                base_version=1,
                instruction="更新",
                content="追加内容",
                base_uuid=str(existing.id),
                base_alias="fact_test",
            ),
            existing_memory=existing,
        )

        [outcome] = await self._process(request)

        assert (outcome.action, outcome.cited_segments) == (GenerationAction.UPDATE, expected)


def _librarian_writes(*drafts: ExtractedMemoryDraft) -> ScriptedLLMService:
    """馆长依次写下每条草稿后结束。"""
    actions = [
        {"tool": "write_memory", "args": draft.model_dump(exclude={"has_value"})}
        for draft in drafts
    ]
    return ScriptedLLMService(
        *(json.dumps(action, ensure_ascii=False) for action in actions),
        json.dumps({"tool": "finish", "args": {}}),
    )


class TestGenerationEngineSettleMode:
    """话题结算按配置的方式分派：一次性生成查重，馆长写下的草稿全部新建。"""

    def setup_method(self):
        self.identity = make_memory_identity_scope()
        self.existing = _make_memory(title="部署改为手动", alias="fact_manual_deploy")
        self.mock_storage = Mock()
        self.mock_storage.search = AsyncMock(
            return_value=[{"memory": self.existing, "score": 0.99}]
        )
        self.mock_storage.list_alias_holders = AsyncMock(return_value=[])
        # 与已有记忆内容相同的草稿：一次性生成的查重判为 IDENTICAL。
        self.same_draft = _make_draft(title="部署改为手动").model_copy(
            update={"content": "旧内容", "source_segments": ["1.1"]}
        )
        self.mock_extractor = AsyncMock()
        self.mock_extractor.extract.return_value = [self.same_draft]

    def _engine(self, config: MemoryGenerationConfig, llm, extractor=None):
        return MemoryGenerationEngine(
            mid_term=self.mock_storage,
            extractor=extractor or self.mock_extractor,
            deduplicator=MemoryDeduplicator(config.deduplicator),
            librarian=create_librarian(
                config,
                mid_term=self.mock_storage,
                llm_service=llm,
                llm_params=LLMRequestParams(model="test-model"),
            ),
        )

    async def _settle(self, engine: MemoryGenerationEngine):
        return await engine.process(
            GenerationRequest(context=_segmented_context()),
            belong_to=self.identity.workspace_identity,
            from_actor=self.identity.actor_identity,
            now=FIXED_NOW,
        )

    @pytest.mark.parametrize(
        ("settle_mode", "expected_action", "is_existing"),
        [("one_shot", GenerationAction.TOUCH, True), ("librarian", GenerationAction.CREATE, False)],
    )
    @pytest.mark.asyncio
    async def test_only_one_shot_settlement_dedups_against_the_library(
        self, settle_mode, expected_action, is_existing
    ):
        """库中有内容相同的记忆：一次性生成只触碰它；馆长写下的草稿照常新建，不提名合并。"""
        config = MemoryGenerationConfig(settle_mode=settle_mode)
        engine = self._engine(config, _librarian_writes(self.same_draft))

        [outcome] = await self._settle(engine)

        assert (outcome.action, outcome.merge_target_id, outcome.atom.id == self.existing.id) == (
            expected_action,
            None,
            is_existing,
        )

    @pytest.mark.asyncio
    async def test_librarian_drafts_cite_segments_of_the_material(self):
        """馆长草稿引用的段换算为事件区间，材料中不存在的标签丢弃。"""
        draft = self.same_draft.model_copy(update={"source_segments": ["2.2", "9.1"]})
        engine = self._engine(
            MemoryGenerationConfig(settle_mode="librarian"), _librarian_writes(draft)
        )

        [outcome] = await self._settle(engine)

        assert outcome.cited_segments == [
            CitedSegment(turn_id="turn-b", first_sequence=3, last_sequence=7)
        ]

    @pytest.mark.asyncio
    async def test_librarian_drafts_of_one_settlement_get_distinct_aliases(self):
        """同一次结算的草稿尚未落库，后一条把前一条已分配的别名视为占用。"""
        first = _make_draft(title="第一条", alias_suffix="deploy_rule")
        second = _make_draft(title="第二条", alias_suffix="deploy_rule")
        engine = self._engine(
            MemoryGenerationConfig(settle_mode="librarian"), _librarian_writes(first, second)
        )

        outcomes = await self._settle(engine)

        assert [outcome.atom.index.alias for outcome in outcomes] == [
            "fact_deploy_rule",
            "fact_deploy_rule_2",
        ]

    @pytest.mark.asyncio
    async def test_librarian_revision_becomes_an_update_of_the_memory_it_read(self):
        """馆长修订读过的记忆：成为以读到的版本为底的 UPDATE，带修改前快照、结算贡献者与
        引用的段；同一次结算的新建不受影响。"""
        self.existing.meta.provenance.contributing_agent_ids = ("creator",)
        self.mock_storage.get_by_alias = AsyncMock(return_value=self.existing)
        draft = _make_draft(title="另一件事", alias_suffix="other_thing")
        llm = ScriptedLLMService(
            json.dumps({"tool": "read_memory", "args": {"alias": "fact_manual_deploy"}}),
            json.dumps(
                {
                    "tool": "revise_memory",
                    "args": {
                        "alias": "fact_manual_deploy",
                        "edits": [_replace("content", "旧内容", "新内容")],
                        "changelog": "部署方式变了",
                        "source_segments": ["2.2"],
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {"tool": "write_memory", "args": draft.model_dump(exclude={"has_value"})},
                ensure_ascii=False,
            ),
            json.dumps({"tool": "finish", "args": {}}),
        )
        engine = self._engine(MemoryGenerationConfig(settle_mode="librarian"), llm)

        created, revised = await self._settle(engine)

        assert (created.action, created.atom.index.title) == (GenerationAction.CREATE, "另一件事")
        assert (revised.action, revised.atom.id, revised.atom.payload.content) == (
            GenerationAction.UPDATE,
            self.existing.id,
            "新内容",
        )
        assert revised.memory_before_snapshot == self.existing
        assert revised.atom.meta.provenance.contributing_agent_ids == ("creator", "a1")
        assert revised.cited_segments == [
            CitedSegment(turn_id="turn-b", first_sequence=3, last_sequence=7)
        ]

    @pytest.mark.parametrize("settle_mode", ["one_shot", "librarian"])
    @pytest.mark.asyncio
    async def test_disabled_extraction_runs_neither_settle_mode(self, settle_mode):
        """extractor.enabled 是两种方式共同的开关：关闭时结算不产生记忆。"""
        config = MemoryGenerationConfig(
            settle_mode=settle_mode, extractor=ExtractorConfig(enabled=False)
        )
        llm = _librarian_writes(self.same_draft)
        extractor = create_extractor(config.extractor, llm, LLMRequestParams(model="test-model"))

        assert await self._settle(self._engine(config, llm, extractor)) == []


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
        provenance = provenance_from_actor(
            identity_scope.actor_identity, _make_context_with_agents(["a1"])
        )

        memory = self.engine._draft_to_memory(
            draft,
            identity_scope.workspace_identity,
            provenance,
            alias="fact_test_alias",
            now=FIXED_NOW,
        )

        assert memory.index.title == "测试标题"
        assert memory.workspace_identity.owner_user_id == "u1"
        assert memory.meta.provenance.source_agent_id == "a1"
        assert memory.meta.provenance.contributing_agent_ids == ("a1",)
        assert memory.meta.lifecycle.confidence_score == 0.9
        # 创建时点 = 提交边界传入的 now（created/updated/decay 同值）
        assert memory.meta.created_at == FIXED_NOW
        assert memory.meta.updated_at == FIXED_NOW
        assert memory.meta.lifecycle.decay_anchor_at == FIXED_NOW

    @pytest.mark.parametrize("memory_type", ["INVALID_TYPE", "CODE_SNIPPET"])
    def test_draft_to_memory_unknown_type(self, memory_type):
        """未知类型名与只作历史取值的旧类型名都落为 FACT，生成不产出旧类型。"""
        draft = _make_draft()
        draft.memory_type = memory_type
        identity_scope = make_memory_identity_scope()

        memory = self.engine._draft_to_memory(
            draft,
            identity_scope.workspace_identity,
            system_settlement_provenance(GenerationContext()),
            alias=None,
            now=FIXED_NOW,
        )

        assert memory.index.memory_type == MemoryType.FACT


class TestMemoryMerge:
    """合并以保留记忆为底：被吸收记忆独有的内容写成对保留记忆的编辑集。"""

    def _engine(self, *outputs) -> MemoryGenerationEngine:
        return MemoryGenerationEngine(
            mid_term=Mock(),
            extractor=LLMMemoryExtractor(
                ExtractorConfig(), ScriptedLLMService(*outputs), LLMRequestParams(model="m")
            ),
            deduplicator=Mock(),
        )

    def _pair(self):
        retained = _make_memory(alias="fact_original")
        retained.index.memory_type = MemoryType.REFLECTION
        retained.meta.provenance.contributing_agent_ids = ("a1", "shared")
        absorbed = _make_memory("新进展")
        absorbed.meta.provenance.contributing_agent_ids = ("shared", "a2")
        return retained, absorbed

    @pytest.mark.asyncio
    async def test_merge_applies_the_edit_set_to_the_retained_memory(self):
        """内容补进保留记忆：标题与类型沿用它，标签只增加，身份与来源不变，贡献者取并集；
        不修改输入。"""
        retained, absorbed = self._pair()
        before = retained.model_copy(deep=True)
        engine = self._engine(
            _edit_set_json(
                {"op": "append", "text": "新进展"}, add_tags=["progress"], changelog="补充进展"
            )
        )

        outcome = await engine.merge(retained, absorbed)

        atom = outcome.atom
        assert outcome.action == GenerationAction.UPDATE
        assert (
            atom.index.title,
            atom.index.memory_type,
            atom.index.tags,
            atom.payload.content,
        ) == (
            before.index.title,
            MemoryType.REFLECTION,
            ["t", "progress"],
            "旧内容\n\n新进展",
        )
        assert (atom.id, atom.index.alias, atom.meta.provenance.source_agent_id) == (
            before.id,
            "fact_original",
            before.meta.provenance.source_agent_id,
        )
        assert atom.meta.provenance.contributing_agent_ids == ("a1", "shared", "a2")
        assert (atom.meta.version, atom.meta.updated_at, atom.meta.lifecycle) == (
            before.meta.version,
            before.meta.updated_at,
            before.meta.lifecycle,
        )
        assert (outcome.changelog, outcome.memory_before_snapshot) == ("补充进展", before)
        assert retained == before

    @pytest.mark.asyncio
    async def test_merge_with_an_empty_edit_set_keeps_the_retained_content(self):
        """被吸收的内容保留记忆都已有：内容与索引不变，仍是一个记下这次吸收的 UPDATE。"""
        retained, absorbed = self._pair()
        engine = self._engine(_edit_set_json(changelog="N 已被 M 覆盖"))

        outcome = await engine.merge(retained, absorbed)

        assert (outcome.action, outcome.changelog) == (GenerationAction.UPDATE, "N 已被 M 覆盖")
        assert (outcome.atom.index, outcome.atom.payload) == (retained.index, retained.payload)
        assert outcome.atom.meta.provenance.contributing_agent_ids == ("a1", "shared", "a2")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output",
        [
            _edit_set_json(_replace("content", "并不存在的原文", "x")),
            json.dumps({"edits": [], "remove_tags": ["t"], "changelog": "去掉标签"}),
        ],
        ids=["invalid", "removes-tags"],
    )
    async def test_merge_rejected_twice_raises_the_rejection(self, output):
        """编辑集带着错误重试一次后仍不能应用，或要删去保留记忆的标签时，抛出拒绝，
        由提交一侧中止合并。"""
        retained, absorbed = self._pair()
        engine = self._engine(output, output)

        with pytest.raises(EditSetError):
            await engine.merge(retained, absorbed)
