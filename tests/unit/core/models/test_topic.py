"""TopicData 原始数据、工作视图派生与构造校验单元测试。

被测对象：``TopicData`` 以交互记录与折叠记录为原始数据并派生工作视图。
保护的契约：
- 工作视图是最后一次折叠之后的交互记录；fold_summary 来自最后一次折叠，没有
  折叠时为空串；total_tokens 是工作视图之和；
- 折叠记录依次拼接必须等于交互记录的前缀，沉淀覆盖的交互数不超过交互记录的长度，
  否则构造失败；
- 判空看是否接收过交互：全部折叠、工作视图为空后仍不为空。
"""

import pytest
from pydantic import ValidationError

from hivememory.core.models import TopicData, TopicFold
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


class TestConstructionValidation:
    def test_folds_must_cover_a_prefix_of_interactions(self):
        """跳过工作视图开头的块去折叠，折叠记录就不再是交互记录的前缀。"""
        b1, b2 = _block("q1"), _block("q2")

        with pytest.raises(ValidationError, match="前缀"):
            _topic(interactions=(b1, b2), folds=(_fold(b2, summary="s"),))

    def test_settled_count_cannot_exceed_interactions(self):
        with pytest.raises(ValidationError, match="沉淀覆盖的交互数"):
            TopicData(
                topic_id="t1",
                workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
                topic_title="title",
                interactions=(_block("q1"),),
                last_update=1.0,
                settled_interaction_count=2,
            )


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
