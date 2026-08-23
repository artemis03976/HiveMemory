"""ShortTermMemoryStore 的 P5 binding 与单写者保护测试。

覆盖设计 v0.6.2 §8 的核心不变量：
- 单一 Store 锁下的冻结读与不可变快照
- ``apply_interaction`` 原子提交 block + 首次 binding + metadata
- 重复使用同一资产只幂等命中既有关系
- PROCESSING/FLUSHING 预约与 busy 拒绝、IDLE/LRU 跳过 busy 候选
- settle/compact/evict 生命周期矩阵对 binding 的保留/清除
"""

import pytest

from hivememory.core.models import (
    BufferState,
    LogicalBlock,
    TopicAssetBinding,
    TurnRecord,
    WorkspaceAssetRef,
    WorkspaceTopicKey,
)
from hivememory.patchouli.errors import TopicBusyError
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from tests.helpers.workspace import make_access_context


def _block(text: str = "q") -> LogicalBlock:
    return LogicalBlock(
        turn=TurnRecord(user_query=text, assistant_final_text="a"),
        total_tokens=10,
    )


def _ref(token: str = "token-1") -> WorkspaceAssetRef:
    return WorkspaceAssetRef(token=token)


class TestApplyInteractionBinding:
    def setup_method(self):
        self.store = ShortTermMemoryStore()
        self.access_context = make_access_context(user_id="u1")
        self.topic = self.store.create_buffer(self.access_context)
        self.key = WorkspaceTopicKey.from_access_context(
            self.access_context, self.topic.topic_id
        )

    def test_apply_interaction_requires_processing_reservation(self):
        with pytest.raises(TopicBusyError):
            self.store.apply_interaction(
                self.access_context,
                self.topic.topic_id,
                "i1",
                _block(),
                asset_id_and_refs=(("asset-1", _ref()),),
            )

    def test_apply_interaction_atomically_commits_block_binding_and_metadata(self):
        assert self.store.reserve_processing(self.key)

        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i1",
            _block("first"),
            asset_id_and_refs=(("asset-1", _ref("token-1")),),
            model_used="model-x",
        )

        data = self.store.get_topic_data(self.access_context, self.topic.topic_id)
        assert data.block_count == 1
        assert data.model_used == "model-x"
        assert len(data.bindings) == 1
        binding = data.bindings[0]
        assert binding.asset_id == "asset-1"
        assert binding.asset_ref.token == "token-1"
        assert binding.first_bound_interaction_id == "i1"

    def test_repeat_asset_use_keeps_first_interaction_and_time(self):
        assert self.store.reserve_processing(self.key)
        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i1",
            _block("first"),
            asset_id_and_refs=(("asset-1", _ref("token-1")),),
        )
        self.store.release_processing(self.key)

        first_binding = self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        )[0]

        # 第二轮再次使用同一资产，只命中既有关系。
        assert self.store.reserve_processing(self.key)
        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i2",
            _block("second"),
            asset_id_and_refs=(("asset-1", _ref("token-1")),),
        )
        self.store.release_processing(self.key)

        bindings = self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        )
        assert len(bindings) == 1
        assert bindings[0].first_bound_interaction_id == "i1"
        assert bindings[0].bound_at == first_binding.bound_at

    def test_distinct_assets_produce_distinct_bindings(self):
        assert self.store.reserve_processing(self.key)
        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i1",
            _block(),
            asset_id_and_refs=(("asset-1", _ref("token-1")), ("asset-2", _ref("token-2"))),
        )

        bindings = self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        )
        assert {b.asset_id for b in bindings} == {"asset-1", "asset-2"}

    def test_binding_requires_interaction_id(self):
        assert self.store.reserve_processing(self.key)
        with pytest.raises(ValueError, match="interaction_id"):
            self.store.apply_interaction(
                self.access_context,
                self.topic.topic_id,
                None,
                _block(),
                asset_id_and_refs=(("asset-1", _ref()),),
            )

    def test_orphan_asset_without_binding_is_legal(self):
        # 不带 asset refs 的 Interaction 不建立任何 binding。
        assert self.store.reserve_processing(self.key)
        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i1",
            _block(),
        )
        assert self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        ) == ()


class TestSingleWriterReservation:
    def setup_method(self):
        self.store = ShortTermMemoryStore()
        self.access_context = make_access_context(user_id="u1")
        self.topic = self.store.create_buffer(self.access_context)
        self.key = WorkspaceTopicKey.from_access_context(
            self.access_context, self.topic.topic_id
        )

    def test_reserve_processing_is_exclusive(self):
        assert self.store.reserve_processing(self.key) is True
        assert self.store.reserve_processing(self.key) is False
        assert self.store.reserve_flushing(self.key) is False
        self.store.release_processing(self.key)
        assert self.store.reserve_flushing(self.key) is True

    def test_release_processing_is_idempotent(self):
        self.store.reserve_processing(self.key)
        self.store.release_processing(self.key)
        self.store.release_processing(self.key)  # 不应抛错
        data = self.store.get_topic_data(self.access_context, self.topic.topic_id)
        assert data.state is BufferState.IDLE

    def test_abort_flushing_restores_idle(self):
        assert self.store.reserve_flushing(self.key)
        self.store.abort_flushing(self.key)
        data = self.store.get_topic_data(self.access_context, self.topic.topic_id)
        assert data.state is BufferState.IDLE
        assert data.topic_id == self.topic.topic_id  # 内容保留

    def test_commit_flushing_evicts_topic(self):
        assert self.store.reserve_flushing(self.key)
        assert self.store.commit_flushing(self.key) is True
        assert self.store.topic_exists(self.access_context, self.topic.topic_id) is False

    def test_freeze_and_evict_rejects_busy_topic(self):
        """automatic settle 必须显式区分 busy，不能把它伪装成正常 skip。"""
        assert self.store.reserve_processing(self.key)
        with pytest.raises(TopicBusyError, match="正忙"):
            self.store.freeze_and_evict(self.key)

        # 拒绝 freeze 后 Topic 仍保留，原预约也没有被越权释放。
        data = self.store.get_topic_data(
            self.access_context,
            self.topic.topic_id,
            touch=False,
        )
        assert data is not None
        assert data.state is BufferState.PROCESSING
        self.store.release_processing(self.key)
        assert self.store.freeze_and_evict(self.key) is not None
        assert self.store.topic_exists(self.access_context, self.topic.topic_id) is False

    def test_get_lru_topic_skips_busy_candidate(self):
        idle = self.store.create_buffer(self.access_context)
        self.store.reserve_processing(self.key)  # 让 self.topic 处于 busy

        lru = self.store.get_lru_topic(self.access_context)
        assert lru == idle.topic_id


class TestBindingLifecycleMatrix:
    def setup_method(self):
        self.store = ShortTermMemoryStore()
        self.access_context = make_access_context(user_id="u1")
        self.topic = self.store.create_buffer(self.access_context)
        self.key = WorkspaceTopicKey.from_access_context(
            self.access_context, self.topic.topic_id
        )

    def _bind(self):
        self.store.reserve_processing(self.key)
        self.store.apply_interaction(
            self.access_context,
            self.topic.topic_id,
            "i1",
            _block(),
            asset_id_and_refs=(("asset-1", _ref()),),
        )
        self.store.release_processing(self.key)

    def test_compact_preserves_binding(self):
        self._bind()
        self.store.reserve_processing(self.key)
        self.store.apply_compaction(self.key, "folded", retain_count=1)
        self.store.release_processing(self.key)

        assert len(
            self.store.list_asset_bindings(self.access_context, self.topic.topic_id)
        ) == 1

    def test_manual_delete_clears_binding(self):
        self._bind()
        self.store.pop_buffer_by_key(self.key)
        assert self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        ) == ()

    def test_freeze_and_evict_clears_binding_but_snapshot_freezes_refs(self):
        self._bind()
        snapshot = self.store.freeze_and_evict(self.key)
        assert snapshot is not None
        assert len(snapshot.bindings) == 1
        # buffer 已随生命周期清除
        assert self.store.list_asset_bindings(
            self.access_context, self.topic.topic_id
        ) == ()
