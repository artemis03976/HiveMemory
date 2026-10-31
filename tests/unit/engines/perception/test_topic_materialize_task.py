"""TopicMaterializeTask.from_topic_data() 转换边界单元测试。

字段映射、Workspace 归属传递、binding 快照和 no-material 判断统一由该类方法
负责；服务与调用方不得重复拼装。结算冻结全部交互与折叠记录，生成材料仍是
与 TopicData 同一派生的工作视图。
"""

from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    TopicFold,
    WorkspaceAssetRef,
)
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from tests.helpers.sessions import make_logical_block
from tests.helpers.workspace import make_identity_scope


def _identity_scope(user_id="u1"):
    return make_identity_scope(user_id=user_id)


def _block(text="q", *, user_id="u1") -> LogicalBlock:
    return make_logical_block(
        text, "a", identity=ActorIdentity(user_id=user_id, agent_id="a1"), total_tokens=len(text)
    )


def _topic_data(scope, *, interactions=(), folds=(), bindings=()):
    return TopicData(
        topic_id="topic-1",
        workspace_identity=scope.workspace_identity,
        topic_title="标题",
        topic_summary="展示摘要",
        interactions=tuple(interactions),
        folds=tuple(folds),
        bindings=tuple(bindings),
        last_update=1.0,
    )


class TestFromTopicData:
    def test_maps_all_fields_from_topic_data(self):
        scope = _identity_scope()
        topic = _topic_data(scope, interactions=(_block("q1"),))

        task = TopicMaterializeTask.from_topic_data(
            topic, belong_to=scope.workspace_identity, reason=TriggerReason.MANUAL_SETTLE
        )

        assert task is not None
        assert task.topic_id == "topic-1"
        assert task.belong_to == scope.workspace_identity
        assert task.topic_title == "标题"
        assert task.topic_summary == "展示摘要"
        assert task.state_summary == ""
        assert task.reason is TriggerReason.MANUAL_SETTLE
        assert [b.user_message for b in task.blocks] == ["q1"]

    def test_workspace_belongs_to_explicit_owner_not_block_actor(self):
        """归属来自调用方显式传入，而不是从 block 内容推导。"""
        scope = _identity_scope(user_id="owner")
        # block 的历史身份与调用方作用域不同（内容事实不参与作用域拼装）。
        topic = _topic_data(scope, interactions=(_block("q", user_id="someone_else"),))

        task = TopicMaterializeTask.from_topic_data(
            topic, belong_to=scope.workspace_identity, reason=TriggerReason.IDLE_TIMEOUT
        )

        assert task.belong_to == scope.workspace_identity
        assert task.belong_to.owner_user_id == "owner"

    def test_empty_topic_returns_none(self):
        scope = _identity_scope()
        topic = _topic_data(scope)

        assert (
            TopicMaterializeTask.from_topic_data(
                topic, belong_to=scope.workspace_identity, reason=TriggerReason.IDLE_TIMEOUT
            )
            is None
        )

    def test_folded_topic_freezes_every_interaction_and_fold(self):
        """结算冻结全部交互与折叠记录；生成材料仍是最后一次折叠的摘要与未折叠的块。"""
        scope = _identity_scope()
        b1, b2, b3 = _block("q1"), _block("q2"), _block("q3")
        fold = TopicFold(
            folded_block_ids=(b1.block_id, b2.block_id), summary="折叠历史", folded_at=1.0
        )
        topic = _topic_data(scope, interactions=(b1, b2, b3), folds=(fold,))

        task = TopicMaterializeTask.from_topic_data(
            topic, belong_to=scope.workspace_identity, reason=TriggerReason.SHUTDOWN
        )

        assert task.interactions == (b1, b2, b3)
        assert task.folds == (fold,)
        assert task.blocks == (b3,)
        assert task.state_summary == "折叠历史"

    def test_bindings_are_frozen_as_snapshot(self):
        from datetime import UTC, datetime

        scope = _identity_scope()
        binding = TopicAssetBinding(
            asset_ref=WorkspaceAssetRef(token="token-1", asset_id="asset-1"),
            first_bound_interaction_id="i1",
            bound_at=datetime.now(UTC),
        )
        topic = _topic_data(scope, interactions=(_block("q"),), bindings=(binding,))

        task = TopicMaterializeTask.from_topic_data(
            topic, belong_to=scope.workspace_identity, reason=TriggerReason.LRU_EVICTION
        )

        assert task.asset_bindings == (binding,)
