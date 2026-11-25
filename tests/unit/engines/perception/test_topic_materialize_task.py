"""TopicMaterializeTask.from_topic_data() 转换边界单元测试。

字段映射、Workspace 归属传递、binding 快照和 no-material 判断统一由该类方法
负责；服务与调用方不得重复拼装。最近一次沉淀之后没有新交互时没有材料；有新交互时
结算冻结全部交互与折叠记录，不只是新增的部分，生成材料仍是与 TopicData 同一派生的
工作视图。
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


def _topic_data(scope, *, interactions=(), folds=(), bindings=(), settled_interaction_count=0):
    return TopicData(
        topic_id="topic-1",
        workspace_identity=scope.workspace_identity,
        topic_title="标题",
        topic_summary="展示摘要",
        interactions=tuple(interactions),
        folds=tuple(folds),
        bindings=tuple(bindings),
        last_update=1.0,
        settled_interaction_count=settled_interaction_count,
    )


class TestFromTopicData:
    def test_maps_all_fields_from_topic_data(self):
        scope = _identity_scope()
        topic = _topic_data(scope, interactions=(_block("q1"),))

        task = TopicMaterializeTask.from_topic_data(
            topic,
            belong_to=scope.workspace_identity,
            reason=TriggerReason.MANUAL_SETTLE,
            retires_topic=True,
        )

        assert task is not None
        assert task.topic_id == "topic-1"
        assert task.belong_to == scope.workspace_identity
        assert task.topic_title == "标题"
        assert task.topic_summary == "展示摘要"
        assert task.fold_summary == ""
        assert task.reason is TriggerReason.MANUAL_SETTLE
        assert task.retires_topic is True
        assert [b.user_message for b in task.blocks] == ["q1"]

    def test_workspace_belongs_to_explicit_owner_not_block_actor(self):
        """归属来自调用方显式传入，而不是从 block 内容推导。"""
        scope = _identity_scope(user_id="owner")
        # block 的历史身份与调用方作用域不同（内容事实不参与作用域拼装）。
        topic = _topic_data(scope, interactions=(_block("q", user_id="someone_else"),))

        task = TopicMaterializeTask.from_topic_data(
            topic,
            belong_to=scope.workspace_identity,
            reason=TriggerReason.IDLE_TIMEOUT,
            retires_topic=False,
        )

        assert task.belong_to == scope.workspace_identity
        assert task.belong_to.owner_user_id == "owner"

    def test_topic_without_unsettled_interactions_returns_none(self):
        """接收过交互、但最近一次沉淀之后没有新交互的话题没有材料。"""
        scope = _identity_scope()
        topic = _topic_data(scope, interactions=(_block("q1"),), settled_interaction_count=1)

        assert (
            TopicMaterializeTask.from_topic_data(
                topic,
                belong_to=scope.workspace_identity,
                reason=TriggerReason.MANUAL_SETTLE,
                retires_topic=True,
            )
            is None
        )

    def test_partially_settled_topic_freezes_every_interaction(self):
        """再次沉淀时材料仍是全部交互，已沉淀的轮次也在其中。"""
        scope = _identity_scope()
        b1, b2 = _block("q1"), _block("q2")
        topic = _topic_data(scope, interactions=(b1, b2), settled_interaction_count=1)

        task = TopicMaterializeTask.from_topic_data(
            topic,
            belong_to=scope.workspace_identity,
            reason=TriggerReason.IDLE_TIMEOUT,
            retires_topic=False,
        )

        assert task.interactions == (b1, b2)
        assert task.blocks == (b1, b2)

    def test_folded_topic_freezes_every_interaction_and_fold(self):
        """结算冻结全部交互与折叠记录；生成材料仍是最后一次折叠的摘要与未折叠的块。"""
        scope = _identity_scope()
        b1, b2, b3 = _block("q1"), _block("q2"), _block("q3")
        fold = TopicFold(
            folded_block_ids=(b1.block_id, b2.block_id), summary="折叠历史", folded_at=1.0
        )
        topic = _topic_data(scope, interactions=(b1, b2, b3), folds=(fold,))

        task = TopicMaterializeTask.from_topic_data(
            topic,
            belong_to=scope.workspace_identity,
            reason=TriggerReason.SHUTDOWN,
            retires_topic=True,
        )

        assert task.interactions == (b1, b2, b3)
        assert task.folds == (fold,)
        assert task.blocks == (b3,)
        assert task.fold_summary == "折叠历史"

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
            topic,
            belong_to=scope.workspace_identity,
            reason=TriggerReason.LRU_EVICTION,
            retires_topic=True,
        )

        assert task.asset_bindings == (binding,)
