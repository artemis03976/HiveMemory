"""TopicData 内容判空语义单元测试。

内容判空契约（topic-content-emptiness-and-manual-lifecycle）：
    has_blocks = len(blocks) > 0
    has_state_summary = bool(state_summary.strip())
    has_content = has_blocks OR has_state_summary
    is_empty = NOT has_content

空白字符串不构成有效 state_summary；has_blocks 保持“是否有原始 block”的窄语义。
"""

import pytest

from hivememory.core.models import LogicalBlock, TopicData, TurnRecord
from tests.helpers.workspace import make_identity_scope


def _make_topic(blocks=None, state_summary=""):
    identity_scope = make_identity_scope(user_id="u1")
    return TopicData(
        topic_id="t1",
        workspace_identity=identity_scope.workspace_identity,
        topic_title="title",
        topic_summary="summary",
        state_summary=state_summary,
        blocks=tuple(blocks or ()),
        last_update=1.0,
    )


def _make_block():
    return LogicalBlock(turn=TurnRecord(user_query="q", assistant_final_text="a"))


class TestTopicDataContentEmptiness:
    """blocks / state_summary 四种组合的内容判空矩阵。"""

    def test_no_blocks_no_summary_is_empty(self):
        topic = _make_topic()
        assert topic.is_empty is True
        assert topic.has_content is False
        assert topic.has_blocks is False

    def test_blocks_without_summary_is_not_empty(self):
        topic = _make_topic(blocks=[_make_block()], state_summary="")
        assert topic.is_empty is False
        assert topic.has_content is True
        assert topic.has_blocks is True

    def test_summary_only_is_not_empty(self):
        """summary-only（折叠历史）不应再被误判为空 Topic。"""
        topic = _make_topic(blocks=[], state_summary="已经折叠的历史内容")
        assert topic.is_empty is False
        assert topic.has_content is True
        assert topic.has_blocks is False

    def test_blocks_and_summary_is_not_empty(self):
        topic = _make_topic(
            blocks=[_make_block()],
            state_summary="折叠摘要与近期工作集并存",
        )
        assert topic.is_empty is False
        assert topic.has_content is True
        assert topic.has_blocks is True

    @pytest.mark.parametrize(
        "summary",
        ["", "   ", "\n\t "],
        ids=["empty", "spaces", "whitespace"],
    )
    def test_blank_summary_does_not_count_as_content(self, summary):
        """空白字符串不构成有效 state_summary。"""
        topic = _make_topic(blocks=[], state_summary=summary)
        assert topic.is_empty is True
        assert topic.has_content is False

    def test_summary_with_blocks_keeps_has_blocks_narrow_semantics(self):
        """has_blocks 不得静默扩大为 has_content。"""
        topic = _make_topic(blocks=[_make_block()], state_summary="summary")
        assert topic.has_blocks is True
        summary_only = _make_topic(blocks=[], state_summary="summary")
        assert summary_only.has_blocks is False
        assert summary_only.has_content is True


class TestTopicDataNoTopicLevelAgentIdentity:
    """话题级 Agent 身份字段的删除守卫（provenance 与责任主体分离）。"""

    def test_current_agent_id_field_is_removed(self):
        """TopicData 不再携带话题级 Agent 身份：来源由 block.turn.identity 记录。

        历史 ``current_agent_id`` 曾以 ``"default"`` 为默认值且无业务消费者，
        容易被误解释为 Memory provenance 来源；该测试防止字段以任何形式复活。
        """
        assert "current_agent_id" not in TopicData.model_fields

    def test_serialized_snapshot_has_no_agent_identity_field(self):
        """TopicData 序列化契约中不出现话题级 Agent 身份字段。"""
        topic = _make_topic(blocks=[_make_block()])

        dumped = topic.model_dump(mode="json")

        assert "current_agent_id" not in dumped
        assert "agent_id" not in dumped
