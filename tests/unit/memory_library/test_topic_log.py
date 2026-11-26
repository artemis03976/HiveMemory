"""话题日志的记录、套用函数与重放的单元测试。

被测对象：``memory_library.topic_log`` 中每种记录的套用函数与 ``replay_topic_log``。
保护的契约：
- 接收一轮只追加交互记录、附件按 ``asset_id`` 幂等绑定；折叠只追加一条折叠记录，不移除交互；
- 套用函数的时间取记录的 ``at``，不取当前时间；
- 沉淀进度只往前，不改写入时间；本话题生成的记忆只改这份记录，合并改指按话题此刻的记录判断；
- 进展记录只往后追加，各条依次覆盖相邻的区间，不改写入时间；从错误的位置开始或覆盖超过交互数时
  构造失败；
- 套用函数以完整构造产生新值，``TopicData`` 的构造校验照常生效；
- 日志必须以唯一的 ``created`` 开头，退役记录由重放单独交出。
"""

import pytest
from pydantic import ValidationError

from hivememory.core.models import TopicData, TopicFold, WorkspaceAssetRef
from hivememory.engines.perception.models import TriggerReason
from hivememory.memory_library.topic_log import (
    TopicCreatedRecord,
    TopicFoldRecord,
    TopicGeneratedMemoriesRecord,
    TopicInteractionRecord,
    TopicProgressEntryRecord,
    TopicRetiredRecord,
    TopicSettledRecord,
    apply_fold,
    apply_generated_memories,
    apply_interaction,
    apply_progress_entry,
    apply_settled,
    decode_topic_record,
    encode_topic_record,
    replay_topic_log,
)
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope

_WORKSPACE = make_identity_scope(user_id="u1").workspace_identity


def _topic(interactions=(), folds=(), settled=0) -> TopicData:
    return TopicData(
        topic_id="t1",
        workspace_identity=_WORKSPACE,
        topic_title="title",
        interactions=tuple(interactions),
        folds=tuple(folds),
        last_update=1.0,
        settled_interaction_count=settled,
    )


def _block(message: str):
    return make_logical_block(message, "a", total_tokens=10)


def _fold(*blocks, summary: str) -> TopicFold:
    return TopicFold(
        folded_block_ids=tuple(block.block_id for block in blocks), summary=summary, folded_at=1.0
    )


def _created(at: float = 1.0) -> TopicCreatedRecord:
    return TopicCreatedRecord(at=at, workspace_identity=_WORKSPACE, topic_title="title")


def _progress_entry(*covered, covered_until: int, at: float = 1.0) -> TopicProgressEntryRecord:
    """覆盖 ``covered`` 这几轮的一条，指回的区间按它们的轮次标识与末事件序号写。"""
    last = covered[-1].turn
    return TopicProgressEntryRecord(
        at=at,
        text=f"entry-{covered_until}",
        from_turn_id=covered[0].turn.turn_id,
        to_turn_id=last.turn_id,
        to_event_seq=last.turn_events[-1].sequence,
        covered_until=covered_until,
    )


class TestApplyFunctions:
    def test_interaction_appends_after_folded_history_and_takes_the_record_time(self):
        """一轮追加到交互记录末尾、进入工作视图；写入时间与模型名取自记录。"""
        b1, b2, b3 = _block("q1"), _block("q2"), _block("q3")
        topic = _topic(interactions=(b1, b2), folds=(_fold(b1, summary="s1"),))

        updated = apply_interaction(
            topic, TopicInteractionRecord(at=42.0, block=b3, model_used="model-a")
        )

        assert updated.interactions == (b1, b2, b3)
        assert updated.folds == topic.folds
        assert updated.blocks == (b2, b3)
        assert (updated.last_update, updated.model_used) == (42.0, "model-a")

    def test_interaction_binds_each_asset_once(self):
        """同一附件只在第一次使用它的交互上绑定，绑定时间取那条记录的时间。"""
        asset = WorkspaceAssetRef(token="token-1", asset_id="asset-1")
        first_at = 1_764_000_000.0
        first = apply_interaction(
            _topic(),
            TopicInteractionRecord(
                at=first_at, block=_block("q1"), asset_refs=(asset,), interaction_id="i-1"
            ),
        )

        second = apply_interaction(
            first,
            TopicInteractionRecord(
                at=first_at + 60, block=_block("q2"), asset_refs=(asset,), interaction_id="i-2"
            ),
        )

        assert [
            (binding.asset_id, binding.first_bound_interaction_id, binding.bound_at.timestamp())
            for binding in second.bindings
        ] == [("asset-1", "i-1", first_at)]

    def test_asset_binding_requires_an_interaction_id(self):
        with pytest.raises(ValidationError, match="interaction_id"):
            TopicInteractionRecord(
                at=1.0,
                block=_block("q1"),
                asset_refs=(WorkspaceAssetRef(token="token-1", asset_id="asset-1"),),
            )

    def test_fold_appends_a_record_and_keeps_every_interaction(self):
        """折叠只追加一条记录并推进工作视图，被折叠的块仍在交互记录中。"""
        b1, b2, b3, b4 = (_block(f"q{i}") for i in range(1, 5))
        topic = _topic(interactions=(b1, b2, b3, b4), folds=(_fold(b1, summary="s1"),))

        folded = apply_fold(
            topic,
            TopicFoldRecord(at=7.0, folded_block_ids=(b2.block_id, b3.block_id), summary="s2"),
        )

        assert folded.interactions == (b1, b2, b3, b4)
        assert folded.folds[-1] == TopicFold(
            folded_block_ids=(b2.block_id, b3.block_id), summary="s2", folded_at=7.0
        )
        assert (folded.blocks, folded.fold_summary, folded.last_update) == ((b4,), "s2", 7.0)

    def test_settled_progress_only_moves_forward_without_touching_the_write_time(self):
        """较早接纳的沉淀后完成时不把进度拉回去；沉淀不是一次写入，写入时间不变。"""
        topic = _topic(interactions=(_block("q1"), _block("q2"), _block("q3")))

        advanced = apply_settled(topic, TopicSettledRecord(at=9.0, interaction_count=3))
        late = apply_settled(advanced, TopicSettledRecord(at=10.0, interaction_count=1))

        assert (late.settled_interaction_count, late.has_unsettled_interactions) == (3, False)
        assert late.last_update == topic.last_update

    def test_writes_change_only_the_record_and_redirect_recorded_merges(self):
        """合并改指只作用于话题此刻记录中的记忆，不在记录中的被吸收记忆不把保留的那条带进来。"""
        topic = apply_generated_memories(
            _topic(interactions=(_block("q1"),), settled=1),
            TopicGeneratedMemoriesRecord(at=2.0, generated=frozenset({"absorbed", "kept"})),
        )

        recorded = apply_generated_memories(
            topic,
            TopicGeneratedMemoriesRecord(
                at=3.0,
                generated=frozenset({"new"}),
                absorbed_into={"absorbed": "retained", "elsewhere": "retained_elsewhere"},
            ),
        )

        assert recorded.topic_generated_memory_ids == {"kept", "retained", "new"}
        assert recorded.model_dump(exclude={"topic_generated_memory_ids"}) == topic.model_dump(
            exclude={"topic_generated_memory_ids"}
        )

    def test_progress_entries_append_over_adjacent_turns_without_touching_the_write_time(self):
        """每一条从上一条覆盖到的位置开始；之后的交互是还没整理的，写入时间不因进展记录变化。"""
        b1, b2, b3, b4 = (_block(f"q{i}") for i in range(1, 5))
        topic = _topic(interactions=(b1, b2, b3, b4))

        first = apply_progress_entry(topic, _progress_entry(b1, b2, covered_until=2, at=5.0))
        second = apply_progress_entry(first, _progress_entry(b3, covered_until=3, at=6.0))

        assert [
            (entry.text, entry.covered_until, entry.written_at) for entry in second.progress_record
        ] == [
            ("entry-2", 2, 5.0),
            ("entry-3", 3, 6.0),
        ]
        assert second.progress_record[0] == first.progress_record[0]
        assert second.recorded_interaction_count == 3
        assert second.to_topic_snapshot().unrecorded_turn_count == 1
        assert second.last_update == topic.last_update

    @pytest.mark.parametrize(
        ("covered", "covered_until"),
        [((2,), 3), ((1, 2), 4)],
        ids=["starts_after_the_recorded_position", "covers_more_than_the_interactions"],
    )
    def test_progress_entry_must_start_at_the_recorded_position_within_the_interactions(
        self, covered, covered_until
    ):
        """已经覆盖了第一轮：跳过第二轮开始的一条，或覆盖到交互记录之外的一条，构造失败。"""
        blocks = (_block("q1"), _block("q2"), _block("q3"))
        topic = apply_progress_entry(
            _topic(interactions=blocks), _progress_entry(blocks[0], covered_until=1)
        )

        with pytest.raises(ValidationError, match="进展记录"):
            apply_progress_entry(
                topic,
                _progress_entry(*(blocks[i] for i in covered), covered_until=covered_until),
            )

    @pytest.mark.parametrize(
        "apply",
        [
            lambda topic, block: apply_interaction(
                topic, TopicInteractionRecord(at=1.0, block=block)
            ),
            lambda topic, block: apply_fold(
                topic, TopicFoldRecord(at=1.0, folded_block_ids=(block.block_id,), summary="s")
            ),
            lambda topic, block: apply_settled(
                topic, TopicSettledRecord(at=1.0, interaction_count=1)
            ),
            lambda topic, block: apply_generated_memories(
                topic, TopicGeneratedMemoriesRecord(at=1.0, generated=frozenset({"m"}))
            ),
            lambda topic, block: apply_progress_entry(
                topic, _progress_entry(block, covered_until=1)
            ),
        ],
        ids=["interaction", "fold", "settled", "generated_memories", "progress_entry"],
    )
    def test_apply_functions_validate_the_new_value(self, apply):
        """套用函数以完整构造产生读取值：错误构造的折叠在下一次套用时被挡住。"""
        b1, b2 = _block("q1"), _block("q2")
        broken = TopicData.model_construct(
            topic_id="t1",
            workspace_identity=_WORKSPACE,
            topic_title="title",
            interactions=(b1, b2),
            folds=(_fold(b2, summary="s"),),
            bindings=(),
            last_update=1.0,
            model_used="",
            settled_interaction_count=0,
            topic_generated_memory_ids=frozenset(),
            progress_record=(),
        )

        with pytest.raises(ValidationError, match="前缀"):
            apply(broken, b1)


class TestReplay:
    def test_replay_rebuilds_the_topic_from_encoded_lines_and_hands_back_retirement(self):
        """编码后的每一行解码回同一条记录；重放依次套用，退役记录单独交出、不改读取值。"""
        b1, b2 = _block("q1"), _block("q2")
        records = [
            _created(at=1.0),
            TopicInteractionRecord(at=2.0, block=b1, model_used="model-a"),
            TopicInteractionRecord(at=3.0, block=b2),
            TopicFoldRecord(at=4.0, folded_block_ids=(b1.block_id,), summary="s1"),
            TopicGeneratedMemoriesRecord(at=5.0, generated=frozenset({"m2", "m1"})),
            _progress_entry(b1, covered_until=1, at=5.5),
            TopicRetiredRecord(at=6.0, reason=TriggerReason.LRU_EVICTION),
            TopicSettledRecord(at=7.0, interaction_count=1),
        ]

        decoded = [decode_topic_record(encode_topic_record(record)) for record in records]
        topic, retired = replay_topic_log("t1", decoded)

        assert decoded == records
        assert retired == records[6]
        assert topic.interactions == (b1, b2)
        assert topic.fold_summary == "s1"
        assert (topic.last_update, topic.model_used) == (4.0, "model-a")
        assert topic.topic_generated_memory_ids == {"m1", "m2"}
        assert topic.settled_interaction_count == 1
        assert [(entry.covered_until, entry.written_at) for entry in topic.progress_record] == [
            (1, 5.5)
        ]

    @pytest.mark.parametrize(
        "records",
        [[], [_created(), _created()]],
        ids=["empty", "created_twice"],
    )
    def test_replay_rejects_a_log_that_does_not_start_with_one_created_record(self, records):
        with pytest.raises(ValueError, match="created"):
            replay_topic_log("t1", records)

    def test_unknown_format_version_cannot_be_decoded(self):
        line = encode_topic_record(_created()).replace(b'"format_version":1', b'"format_version":2')

        with pytest.raises(ValueError, match="format_version"):
            decode_topic_record(line)
