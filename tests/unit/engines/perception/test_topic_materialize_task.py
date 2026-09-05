"""TopicMaterializeTask.from_topic_data() 转换边界单元测试。

字段映射、Workspace/执行者作用域传递、binding 快照、worth_saving 过滤和
no-material 判断统一由该类方法负责；服务与调用方不得重复拼装。
"""

import pytest

from hivememory.core.models import (
    Identity,
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    TurnRecord,
    WorkspaceAssetRef,
)
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1"):
    return make_identity_scope(user_id=user_id)


def _block(text="q", *, worth_saving=None, user_id="u1") -> LogicalBlock:
    turn = TurnRecord(
        identity=Identity(user_id=user_id, agent_id="a1"),
        user_query=text,
        assistant_final_text="a",
    )
    return LogicalBlock(turn=turn, total_tokens=len(text), worth_saving=worth_saving)


def _topic_data(scope, *, blocks=(), bindings=(), state_summary=""):
    return TopicData(
        topic_id="topic-1",
        workspace_identity=scope.workspace_identity,
        topic_title="标题",
        topic_summary="展示摘要",
        state_summary=state_summary,
        blocks=tuple(blocks),
        bindings=tuple(bindings),
        last_update=1.0,
    )


class TestFromTopicData:
    def test_maps_all_fields_from_topic_data(self):
        scope = _identity_scope()
        topic = _topic_data(scope, blocks=(_block("q1"),))

        task = TopicMaterializeTask.from_topic_data(
            topic, identity_scope=scope, reason=TriggerReason.MANUAL_SETTLE
        )

        assert task is not None
        assert task.topic_id == "topic-1"
        assert task.identity_scope is scope
        assert task.workspace_identity == scope.workspace_identity
        assert task.topic_title == "标题"
        assert task.topic_summary == "展示摘要"
        assert task.state_summary == ""
        assert task.reason is TriggerReason.MANUAL_SETTLE
        assert [b.turn.user_query for b in task.blocks] == ["q1"]

    def test_identity_scope_passed_through_not_derived(self):
        """执行者作用域来自调用方显式传入，而不是从 block 内容推导。

        历史 ``_build_settle_payload`` 会取最后一个 worth-saving block 的身份
        拼装作用域；该行为已删除，此测试防止回归。
        """
        scope = _identity_scope(user_id="owner")
        # block 的历史身份与调用方作用域不同（内容事实不参与作用域拼装）。
        topic = _topic_data(scope, blocks=(_block("q", user_id="someone_else"),))

        task = TopicMaterializeTask.from_topic_data(
            topic, identity_scope=scope, reason=TriggerReason.IDLE_TIMEOUT
        )

        assert task.identity_scope == scope
        assert task.identity_scope.actor_identity.user_id == "owner"
        assert task.workspace_identity == scope.workspace_identity

    def test_worth_saving_false_blocks_are_filtered(self):
        scope = _identity_scope()
        topic = _topic_data(
            scope,
            blocks=(
                _block("keep", worth_saving=None),
                _block("drop", worth_saving=False),
            ),
        )

        task = TopicMaterializeTask.from_topic_data(
            topic, identity_scope=scope, reason=TriggerReason.SHUTDOWN
        )

        assert [b.turn.user_query for b in task.blocks] == ["keep"]

    def test_no_material_returns_none_when_all_blocks_filtered(self):
        scope = _identity_scope()
        topic = _topic_data(scope, blocks=(_block("drop", worth_saving=False),))

        assert (
            TopicMaterializeTask.from_topic_data(
                topic, identity_scope=scope, reason=TriggerReason.MANUAL_SETTLE
            )
            is None
        )

    def test_empty_topic_returns_none(self):
        scope = _identity_scope()
        topic = _topic_data(scope, blocks=())

        assert (
            TopicMaterializeTask.from_topic_data(
                topic, identity_scope=scope, reason=TriggerReason.IDLE_TIMEOUT
            )
            is None
        )

    def test_bindings_are_frozen_as_snapshot(self):
        from datetime import datetime

        scope = _identity_scope()
        binding = TopicAssetBinding(
            asset_id="asset-1",
            asset_ref=WorkspaceAssetRef(token="token-1"),
            first_bound_interaction_id="i1",
            bound_at=datetime.now(),
        )
        topic = _topic_data(scope, blocks=(_block("q"),), bindings=(binding,))

        task = TopicMaterializeTask.from_topic_data(
            topic, identity_scope=scope, reason=TriggerReason.LRU_EVICTION
        )

        assert task.asset_bindings == (binding,)

    def test_state_summary_travels_with_task(self):
        scope = _identity_scope()
        topic = _topic_data(scope, blocks=(_block("q"),), state_summary="折叠历史")

        task = TopicMaterializeTask.from_topic_data(
            topic, identity_scope=scope, reason=TriggerReason.IDLE_TIMEOUT
        )

        assert task.state_summary == "折叠历史"
