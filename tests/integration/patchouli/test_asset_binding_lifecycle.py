"""TopicAssetBinding 与 WorkspaceAssetStore remove/acquire 竞态集成测试。

使用真实 ``InMemoryWorkspaceAssetStore``、``ShortTermMemoryStore`` 与
``PerceptionFamiliar`` 验证 design v0.6.2 §8.6 的跨 Store 竞态结果：
- remove-before-acquire：本轮无使用事实，不建立 binding；
- acquire-before-remove：已有 lease 可完成，成功 Interaction 后留下 binding；
- remove 后历史 binding 保留，但新 acquire 稳定失败。
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.errors import AssetRemovedError
from hivememory.core.models import (
    AssetRepresentationKind,
    TurnEvent,
    WorkspaceAssetMetadata,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.patchouli.services.perception import PerceptionFamiliar
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.system.runtime.workspace.store import InMemoryWorkspaceAssetStore
from tests.helpers.workspace import make_identity_scope


def _make_ready_asset(asset_store: InMemoryWorkspaceAssetStore, scope):
    """构造一个 READY 的 RAW 资产并返回 (asset, ref)。"""
    handle = asset_store.create_asset(
        scope,
        WorkspaceAssetMetadata(
            kind="image",
            display_name="diagram",
            media_type="image/png",
            size_bytes=4,
            required_representation_kind=AssetRepresentationKind.RAW,
        ),
        client_operation_id="op-1",
    )
    asset = asset_store.register_raw_representation(
        scope,
        handle.asset_ref,
        content_object=b"data",
        content_hash="hash",
        producer="test",
        producer_version="1",
    )
    return asset, handle.asset_ref


def _make_familiar_with_store():
    """组装真实 Familiar + Store（binding 写入走 apply_interaction 真实链路）。"""
    store = ShortTermMemoryStore()
    familiar = PerceptionFamiliar(
        engine=MemoryPerceptionEngine(
            config=SemanticFlowPerceptionConfig(fold_token_threshold=999999),
            relay_controller=Mock(),
        ),
        store=store,
        working_set=TopicWorkingSet(),
        bus=Mock(request=AsyncMock(return_value=None)),
        config=SimpleNamespace(idle_timeout_seconds=900),
        interaction_journal=InMemoryInteractionApplyJournal(),
    )
    return familiar, store


def _payload() -> InteractionPayload:
    return InteractionPayload(
        user_message="q",
        assistant_final_text="a",
        turn_events=[
            TurnEvent(
                kind="assistant_message",
                sequence=0,
                role="assistant",
                content="a",
            )
        ],
    )


def test_remove_before_acquire_establishes_no_binding():
    asset_store = InMemoryWorkspaceAssetStore()
    _, store = _make_familiar_with_store()
    scope = make_identity_scope(user_id="u1")
    asset, ref = _make_ready_asset(asset_store, scope)

    topic = store.create(scope)
    asset_store.remove_asset(scope, ref)

    # remove 后 acquire 失败 -> 本轮没有使用事实，也就无从建立 binding。
    with pytest.raises(AssetRemovedError):
        asset_store.acquire_ready_representation(scope, ref)

    assert asset.asset_id
    assert store.get(scope, topic.topic_id).bindings == ()


@pytest.mark.asyncio
async def test_acquire_before_remove_leaves_binding_and_blocks_future_acquire():
    asset_store = InMemoryWorkspaceAssetStore()
    familiar, store = _make_familiar_with_store()
    scope = make_identity_scope(user_id="u1")
    asset, ref = _make_ready_asset(asset_store, scope)

    topic = store.create(scope)

    # acquire 先于 remove：已有 lease 可完成本轮使用。
    lease = asset_store.acquire_ready_representation(scope, ref)
    assert lease.representation.asset_id == asset.asset_id

    # remove 发生在 acquire 之后；既有 lease 仍有效，但后续 acquire 被拒绝。
    asset_store.remove_asset(scope, ref)

    # 成功 Interaction 在 remove 后提交 binding：记录 remove 前已发生的真实使用。
    await familiar.apply_interaction(
        _payload(),
        identity_scope=scope,
        target_topic_id=topic.topic_id,
        interaction_id="i1",
        asset_id_and_refs=((asset.asset_id, ref),),
    )

    bindings = store.get(scope, topic.topic_id).bindings
    assert len(bindings) == 1
    assert bindings[0].asset_id == asset.asset_id

    # 历史 binding 不恢复当前可用性；新 acquire 仍稳定失败。
    with pytest.raises(AssetRemovedError):
        asset_store.acquire_ready_representation(scope, ref)


@pytest.mark.asyncio
async def test_commit_before_remove_preserves_binding():
    asset_store = InMemoryWorkspaceAssetStore()
    familiar, store = _make_familiar_with_store()
    scope = make_identity_scope(user_id="u1")
    asset, ref = _make_ready_asset(asset_store, scope)

    topic = store.create(scope)

    lease = asset_store.acquire_ready_representation(scope, ref)
    await familiar.apply_interaction(
        _payload(),
        identity_scope=scope,
        target_topic_id=topic.topic_id,
        interaction_id="i1",
        asset_id_and_refs=((asset.asset_id, ref),),
    )

    asset_store.remove_asset(scope, ref)

    # remove 只终止后续可用性，不改写历史使用事实。
    bindings = store.get(scope, topic.topic_id).bindings
    assert len(bindings) == 1
    assert lease.asset_ref.token == ref.token
