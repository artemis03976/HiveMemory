"""TopicData 原始数据、工作视图派生与写函数单元测试。

被测对象：``TopicData`` 以交互记录与折叠记录为原始数据并派生工作视图；
``merge_interaction_into_topic``、``fold_topic``、``mark_topic_settled`` 与
``record_topic_writes`` 是仅有的写函数。
保护的契约：
- 工作视图是最后一次折叠之后的交互记录；fold_summary 来自最后一次折叠，没有
  折叠时为空串；total_tokens 是工作视图之和；
- 写入一轮只追加交互记录；折叠追加一条折叠记录，不移除交互记录；
- 折叠记录依次拼接必须等于交互记录的前缀，否则构造失败；写函数都以完整
  构造产生快照，这一校验不会被绕过；
- 沉淀进度只记下覆盖到的交互数，不改内容与写入时间；之后写入的交互是未沉淀的；
- 写入记录只改这份记录，合并改指按话题当前的记录判断；
- 话题以 JSON 往返后与原快照相等，读回关闭时保存的话题池不另设序列化；
- 判空看是否接收过交互：全部折叠、工作视图为空后仍不为空。
"""

import json

import pytest
from pydantic import ValidationError

from hivememory.core.models import (
    TopicData,
    TopicFold,
    WorkspaceAssetRef,
    fold_topic,
    mark_topic_settled,
    merge_interaction_into_topic,
    record_topic_writes,
)
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope


def _topic(interactions=(), folds=()) -> TopicData:
    return TopicData(
        topic_id="t1",
        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
        topic_title="title",
        interactions=tuple(interactions),
        folds=tuple(folds),
        last_update=1.0,
    )


def _block(message: str, total_tokens: int = 10):
    return make_logical_block(message, "a", total_tokens=total_tokens)


def _fold(*blocks, summary: str) -> TopicFold:
    return TopicFold(
        folded_block_ids=tuple(block.block_id for block in blocks),
        summary=summary,
        folded_at=1.0,
    )


class TestWorkingView:
    def test_working_view_follows_the_last_fold(self):
        """工作视图从最后一次折叠之后开始，摘要取最后一次折叠，token 只统计工作视图。"""
        b1, b2, b3, b4 = (_block(f"q{i}", total_tokens=10 * i) for i in range(1, 5))

        topic = _topic(
            interactions=(b1, b2, b3, b4),
            folds=(_fold(b1, summary="s1"), _fold(b2, summary="s2")),
        )

        assert topic.blocks == (b3, b4)
        assert topic.fold_summary == "s2"
        assert topic.total_tokens == 70

    def test_topic_without_folds_works_on_every_interaction(self):
        b1, b2 = _block("q1", total_tokens=5), _block("q2", total_tokens=7)

        topic = _topic(interactions=(b1, b2))

        assert topic.blocks == (b1, b2)
        assert topic.fold_summary == ""
        assert topic.total_tokens == 12


class TestWriteFunctions:
    def test_merge_appends_interaction_after_folded_history(self):
        """写入一轮只追加交互记录：已有折叠不变，新块进入工作视图末尾。"""
        b1, b2, b3 = _block("q1"), _block("q2"), _block("q3")
        topic = _topic(interactions=(b1, b2), folds=(_fold(b1, summary="s1"),))

        merged = merge_interaction_into_topic(topic, b3, (), None, None)

        assert merged.interactions == (b1, b2, b3)
        assert merged.folds == topic.folds
        assert merged.blocks == (b2, b3)

    def test_fold_appends_record_and_keeps_every_interaction(self):
        """折叠只追加一条记录并推进工作视图，被折叠的块仍在交互记录中。"""
        b1, b2, b3, b4 = (_block(f"q{i}") for i in range(1, 5))
        topic = _topic(interactions=(b1, b2, b3, b4), folds=(_fold(b1, summary="s1"),))

        folded = fold_topic(topic, [b2, b3], "s2")

        assert folded.interactions == (b1, b2, b3, b4)
        assert [fold.folded_block_ids for fold in folded.folds] == [
            (b1.block_id,),
            (b2.block_id, b3.block_id),
        ]
        assert folded.blocks == (b4,)
        assert folded.fold_summary == "s2"

    def test_folds_must_cover_a_prefix_of_interactions(self):
        """跳过工作视图开头的块去折叠，折叠记录就不再是交互记录的前缀。"""
        b1, b2 = _block("q1"), _block("q2")

        with pytest.raises(ValidationError, match="前缀"):
            _topic(interactions=(b1, b2), folds=(_fold(b2, summary="s"),))

    @pytest.mark.parametrize(
        "write",
        [
            lambda topic, block: merge_interaction_into_topic(topic, block, (), None, None),
            lambda topic, block: fold_topic(topic, [block], "s"),
            lambda topic, block: mark_topic_settled(topic, 1),
            lambda topic, block: record_topic_writes(topic, frozenset({"m"}), {}),
        ],
        ids=["merge", "fold", "mark_settled", "record_writes"],
    )
    def test_write_functions_validate_the_new_snapshot(self, write):
        """写函数以完整构造产生快照：写函数之外的错误构造在下一次写入时被挡住。"""
        b1, b2 = _block("q1"), _block("q2")
        broken = TopicData.model_construct(
            topic_id="t1",
            workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
            topic_title="title",
            interactions=(b1, b2),
            folds=(_fold(b2, summary="s"),),
            last_update=1.0,
        )

        with pytest.raises(ValidationError, match="前缀"):
            write(broken, b1)


class TestSettlementProgress:
    def test_interaction_after_settlement_is_unsettled(self):
        """沉淀只记下覆盖到的交互数、不改写入时间；此后写入的交互又是未沉淀的。"""
        b1, b2 = _block("q1"), _block("q2")
        topic = _topic(interactions=(b1,))

        settled = mark_topic_settled(topic, 1)
        continued = merge_interaction_into_topic(settled, b2, (), None, None)

        assert settled.interactions == (b1,)
        assert settled.last_update == topic.last_update
        assert settled.has_unsettled_interactions is False
        assert continued.settled_interaction_count == 1
        assert continued.has_unsettled_interactions is True

    def test_settled_count_cannot_exceed_interactions(self):
        with pytest.raises(ValidationError, match="沉淀覆盖的交互数"):
            mark_topic_settled(_topic(interactions=(_block("q1"),)), 2)


class TestTopicWrites:
    def test_record_writes_changes_only_the_record_and_redirects_recorded_merges(self):
        """记下写入只改写入记录；合并改指只作用于话题当前记录中的记忆，不在记录中的被吸收
        记忆不把保留的那条带进来。"""
        b1 = _block("q1")
        topic = mark_topic_settled(_topic(interactions=(b1,)), 1)
        topic = record_topic_writes(topic, frozenset({"absorbed", "kept"}), {})

        recorded = record_topic_writes(
            topic,
            frozenset({"new"}),
            {"absorbed": "retained", "elsewhere": "retained_elsewhere"},
        )

        assert recorded.topic_write_memory_ids == {"kept", "retained", "new"}
        assert recorded.model_dump(exclude={"topic_write_memory_ids"}) == topic.model_dump(
            exclude={"topic_write_memory_ids"}
        )

    def test_topic_survives_a_json_round_trip(self):
        """交互记录、折叠记录、资产绑定、沉淀进度与写入记录经 JSON 往返后与原快照相等。"""
        b1, b2 = _block("q1"), _block("q2")
        topic = merge_interaction_into_topic(
            _topic(interactions=(b1,)),
            b2,
            (WorkspaceAssetRef(token="token-1", asset_id="asset-1"),),
            "i-2",
            "model-a",
        )
        topic = mark_topic_settled(fold_topic(topic, [b1], "s1"), 1)
        topic = record_topic_writes(topic, frozenset({"m2", "m1"}), {})

        restored = TopicData.model_validate(json.loads(topic.model_dump_json()))

        assert restored == topic


class TestContentEmptiness:
    def test_topic_without_interactions_is_empty(self):
        topic = _topic()

        assert topic.is_empty is True
        assert topic.has_content is False

    def test_fully_folded_topic_keeps_content_without_working_blocks(self):
        """交互全部折叠后工作视图为空，但话题接收过交互，仍不为空。"""
        b1 = _block("q1")

        topic = _topic(interactions=(b1,), folds=(_fold(b1, summary="s1"),))

        assert topic.blocks == ()
        assert topic.has_content is True
        assert topic.is_empty is False
