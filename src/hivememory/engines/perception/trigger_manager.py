"""Perception-owned Topic lifecycle and compaction orchestration."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING

from hivememory.core.models import (
    BufferState,
    IdentityScope,
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    WorkspaceAssetRef,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.patchouli.errors import TopicBusyError

from hivememory.engines.perception.models import (
    AutomaticSettleResult,
    FlushEvent,
    FlushReason,
    TopicMaterializeTask,
)

if TYPE_CHECKING:
    from hivememory.engines.perception.relay_controller import BaseRelayController
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


# ========== 决策矩阵：触发原因 -> 原子操作组合 ==========
DECISION_MATRIX: Dict[FlushReason, Dict[str, bool]] = {
    FlushReason.TOKEN_OVERFLOW: {
        "settle": False,  # 话题未完，暂不结算
        "compact": True,  # 必须留摘要以接力
        "evict": False,   # 保持存活
    },
    FlushReason.IDLE_TIMEOUT: {
        "settle": True,   # 完整事件，落库
        "compact": False, # 无需摘要，人已走
        "evict": True,    # 踢出内存
    },
    FlushReason.LRU_EVICTION: {
        "settle": True,   # 完整事件，落库
        "compact": False,
        "evict": True,    # 踢出内存
    },
    FlushReason.SHUTDOWN: {
        "settle": True,   # 进程关闭前强制结算
        "compact": False,
        "evict": True,    # 清空内存态话题
    },
    FlushReason.MANUAL_SETTLE: {
        "settle": True,   # 冻结材料并结算为记忆资产
        "compact": False, # 手动 settle 不再 compact
        "evict": True,    # 接纳成功后结束 Topic 生命周期
    },
    FlushReason.MANUAL_COMPACT: {
        "settle": False,  # 只压缩工作集
        "compact": True,  # 合并 previous summary 与旧前缀
        "evict": False,   # 保留 Topic
    },
    FlushReason.MANUAL_DELETE: {
        "settle": False,  # 不构造 settlement
        "compact": False, # 不生成摘要
        "evict": True,    # 丢弃 Topic
    },
}


class TriggerManager:
    """Own Topic state transitions and lifecycle decisions for Perception.

    It persists immutable snapshots through ``ShortTermMemoryStore``.  No
    storage key is constructed here; targets are always an access scope and a
    globally unique topic ID.
    """

    def __init__(
        self,
        store: "ShortTermMemoryStore",
        relay_controller: "BaseRelayController",
        *,
        lock: threading.RLock | None = None,
    ) -> None:
        """
        初始化 TriggerManager

        Args:
            store: ShortTermMemoryStore 实例（短期记忆存储）
            relay_controller: RelayController 实例（用于生成摘要）
        """
        self._store = store
        self._relay_controller = relay_controller
        self._lock = lock or threading.RLock()

    async def resolve_topic(
        self,
        trigger: FlushEvent,
        *,
        retain_recent_blocks: int | None = None,
    ) -> TopicMaterializeTask | None:
        actions = DECISION_MATRIX.get(trigger.reason)
        if actions is None:
            logger.error("未知的触发原因: %s", trigger.reason)
            return None
        if trigger.reason is FlushReason.MANUAL_SETTLE:
            raise ValueError("MANUAL_SETTLE 必须通过 prepare_manual_settle 处理")
        if actions["settle"] and actions["evict"]:
            return self.settle_and_evict(
                trigger.identity_scope,
                trigger.topic_id,
                trigger.reason,
            ).settlement
        if actions["compact"]:
            return await self._compact_path(trigger, retain_recent_blocks)
        if actions["evict"]:
            self.delete_if_idle(trigger.identity_scope, trigger.topic_id)
        return None

    def settle_and_evict(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        reason: FlushReason,
    ) -> AutomaticSettleResult:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None:
                return AutomaticSettleResult(evicted=False)
            if snapshot.state is not BufferState.IDLE:
                raise TopicBusyError(f"topic '{topic_id}' 正忙，无法执行结算")
            if not self._store.delete(identity_scope, topic_id):
                return AutomaticSettleResult(evicted=False)
        if snapshot.is_empty:
            logger.debug("settle_and_evict: topic 内容为空，已驱逐但无可结算材料")
            return AutomaticSettleResult(evicted=True)
        return AutomaticSettleResult(
            evicted=True,
            settlement=self._build_settle_payload_from_snapshot(snapshot, reason),
        )

    def prepare_manual_settle(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicMaterializeTask | None:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None:
                raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
            if snapshot.state is not BufferState.IDLE:
                raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始 manual settle")
            flushing = snapshot.model_copy(
                update={"state": BufferState.FLUSHING, "last_update": datetime.now().timestamp()}
            )
            self._store.put(flushing)
        if flushing.is_empty:
            return None
        return self._build_settle_payload_from_snapshot(flushing, FlushReason.MANUAL_SETTLE)

    def commit_manual_settle(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None or snapshot.state is not BufferState.FLUSHING:
                return False
            return self._store.delete(identity_scope, topic_id)

    def abort_manual_settle(self, identity_scope: IdentityScope, topic_id: str) -> None:
        self._set_state(identity_scope, topic_id, BufferState.FLUSHING, BufferState.IDLE)

    def reserve_processing(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None or snapshot.state is not BufferState.IDLE:
                return False
            self._store.put(
                snapshot.model_copy(
                    update={
                        "state": BufferState.PROCESSING,
                        "last_update": datetime.now().timestamp(),
                    }
                )
            )
            return True

    def release_processing(self, identity_scope: IdentityScope, topic_id: str) -> None:
        self._set_state(identity_scope, topic_id, BufferState.PROCESSING, BufferState.IDLE)

    def apply_interaction(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        interaction_id: str | None,
        block: LogicalBlock,
        *,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
        model_used: str | None = None,
    ) -> TopicData:
        """Apply one interaction to a PROCESSING Topic in the domain layer."""
        if asset_id_and_refs and not interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None:
                raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
            if snapshot.state is not BufferState.PROCESSING:
                raise TopicBusyError(f"topic '{topic_id}' 未持有 PROCESSING 预约")
            bindings = list(snapshot.bindings)
            existing_ids = {binding.asset_id for binding in bindings}
            now = datetime.now()
            for asset_id, asset_ref in asset_id_and_refs:
                if not isinstance(asset_id, str) or not asset_id.strip():
                    raise ValueError("asset_id 不能为空")
                if not isinstance(asset_ref, WorkspaceAssetRef):
                    raise TypeError("asset_ref 必须是 WorkspaceAssetRef")
                if asset_id in existing_ids:
                    continue
                bindings.append(
                    TopicAssetBinding(
                        asset_id=asset_id.strip(),
                        asset_ref=asset_ref,
                        first_bound_interaction_id=interaction_id,
                        bound_at=now,
                    )
                )
                existing_ids.add(asset_id)
            updated = snapshot.model_copy(
                update={
                    "blocks": (*snapshot.blocks, block),
                    "bindings": tuple(bindings),
                    "total_tokens": snapshot.total_tokens + block.total_tokens,
                    "last_update": now.timestamp(),
                    "model_used": model_used or snapshot.model_used,
                }
            )
            self._store.put(updated)
            return updated

    async def _compact_path(
        self,
        trigger: FlushEvent,
        retain_recent_blocks: int | None,
    ) -> TopicMaterializeTask | None:
        if retain_recent_blocks is None:
            raise ValueError(f"{trigger.reason.value} requires retain_recent_blocks")
        if retain_recent_blocks < 1:
            raise ValueError("retain_recent_blocks must be >= 1")
        scope = trigger.identity_scope
        topic_id = trigger.topic_id
        manual = trigger.reason is FlushReason.MANUAL_COMPACT
        reserved_manual = False
        try:
            with self._lock:
                snapshot = self._store.get(scope, topic_id, touch=False)
                if manual:
                    if snapshot is None:
                        raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
                    if snapshot.state is not BufferState.IDLE:
                        raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始 manual compact")
                    self._store.put(snapshot.model_copy(update={"state": BufferState.PROCESSING}))
                    reserved_manual = True
                    snapshot = self._store.get(scope, topic_id, touch=False)
                elif snapshot is None:
                    return None
                if snapshot is None or snapshot.is_empty:
                    return None
                blocks = list(snapshot.blocks)
                fold_count = max(0, len(blocks) - retain_recent_blocks)
                if fold_count == 0:
                    return None
                summary = self._relay_controller.generate_summary(
                    blocks_to_fold=blocks[:fold_count],
                    previous_summary=snapshot.state_summary,
                )
                retained = tuple(blocks[-retain_recent_blocks:])
                updated = snapshot.model_copy(
                    update={
                        "state_summary": summary,
                        "blocks": retained,
                        "total_tokens": sum(block.total_tokens for block in retained),
                        "last_update": datetime.now().timestamp(),
                        "state": BufferState.IDLE if manual else snapshot.state,
                    }
                )
                self._store.put(updated)
            return None
        finally:
            if reserved_manual:
                with self._lock:
                    current = self._store.get(scope, topic_id, touch=False)
                    if current is not None and current.state is BufferState.PROCESSING:
                        self._store.put(
                            current.model_copy(
                                update={
                                    "state": BufferState.IDLE,
                                    "last_update": datetime.now().timestamp(),
                                }
                            )
                        )

    def delete_if_idle(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is None:
                return False
            if snapshot.state is not BufferState.IDLE:
                raise TopicBusyError(f"topic '{topic_id}' 正忙，无法删除")
            return self._store.delete(identity_scope, topic_id)

    def _set_state(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        expected: BufferState,
        target: BufferState,
    ) -> None:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id, touch=False)
            if snapshot is not None and snapshot.state is expected:
                self._store.put(
                    snapshot.model_copy(
                        update={
                            "state": target,
                            "last_update": datetime.now().timestamp(),
                        }
                    )
                )

    def _build_settle_payload_from_snapshot(
        self,
        snapshot: TopicData,
        reason: FlushReason,
    ) -> TopicMaterializeTask | None:
        return self._build_settle_payload(
            topic_id=snapshot.topic_id,
            blocks_snapshot=list(snapshot.blocks),
            state_summary=snapshot.state_summary,
            reason=reason,
            workspace_identity=snapshot.workspace_identity,
            topic_title=snapshot.topic_title,
            topic_summary=snapshot.topic_summary,
            asset_bindings=snapshot.bindings,
        )

    def _build_settle_payload(
        self,
        topic_id: str,
        blocks_snapshot: list[LogicalBlock],
        state_summary: str,
        reason: FlushReason,
        workspace_identity: WorkspaceIdentity,
        topic_title: str = "",
        topic_summary: str = "",
        asset_bindings: tuple = (),
    ) -> TopicMaterializeTask | None:
        blocks_to_settle = [block for block in blocks_snapshot if block.worth_saving is not False]
        if not blocks_to_settle:
            logger.debug("所有 blocks 被 worth_saving 过滤，跳过 Settle")
            return None

        return TopicMaterializeTask(
            topic_id=topic_id,
            identity_scope=IdentityScope(
                actor_identity=blocks_to_settle[-1].identity,
                workspace_identity=workspace_identity,
            ),
            topic_title=topic_title,
            topic_summary=topic_summary,
            blocks=blocks_to_settle,
            state_summary=state_summary,
            asset_bindings=tuple(asset_bindings),
            reason=reason,
        )


__all__ = ["TriggerManager", "DECISION_MATRIX"]
