"""
MemoryLibrary 三层 Store

每层 Store 持有对应的 StoragePort，封装上层调度逻辑。

ShortTermMemoryStore:
    - 持有 InMemoryShortTermStorage（Phase 1），future 可替换为 RedisShortTermStorage
    - 承接短期话题调度方法（apply_interaction / apply_compaction / LRU 等）
    - 所有 buffer 字段写操作必须通过命名方法，不允许调用方直接写字段
    - P5 起以单一 RLock 作为 SemanticBuffer 的唯一同步与变更边界，公开读取只返回
      冻结的 TopicData 快照，SemanticBuffer 不再越过 Store 边界

MidTermMemoryStore:
    - 持有 primary（向量库）和 optional secondary（图库等）Port
    - 写入时同步到所有后端

LongTermMemoryStore:
    - 持有 LongTermStoragePort
    - 不负责跨层状态转移，由 MemoryLibrary 编排

实现阶段: Phase 1
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from hivememory.core.models import (
    BufferState,
    LogicalBlock,
    MemoryAtom,
    TopicAssetBinding,
    TopicData,
    IdentityScope,
    WorkspaceAssetRef,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
    WorkspaceTopicKey,
    require_identity_scope,
)
from hivememory.core.models.artifact import ArtifactRef
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.errors import TopicBusyError
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.buffer import SemanticBuffer
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import (
    ArtifactStoragePort,
    LongTermStoragePort,
    MidTermStoragePort,
    ShortTermStoragePort,
)

logger = logging.getLogger(__name__)


# ============ ShortTermMemoryStore ============

class ShortTermMemoryStore:
    """
    短期记忆存储（MMU）

    持有 ShortTermStoragePort 实现，提供 buffer CRUD 与上层调度方法。
    Port contract is synchronous because the perception layer uses this store on
    its hot path without await points.

    P5 起 ``SemanticBuffer`` 的全部字段、Topic pool 与 last-active 索引都由本
    Store 的单一 ``threading.RLock`` 保护；公开读取只在锁内构造冻结快照，任何
    方法都不再向调用方泄漏可变 buffer。同一 Topic 的写入通过
    ``PROCESSING``/``FLUSHING`` 状态预约串行化，不引入 per-topic lock 或 CAS。
    """

    def __init__(
        self,
        port: Optional[ShortTermStoragePort] = None,
        max_resident_topics: int = 5,
    ) -> None:
        self._port: ShortTermStoragePort = port or InMemoryShortTermStorage()
        self.max_resident_topics = max_resident_topics
        self._last_active_topic_keys: dict[tuple[str, str], WorkspaceTopicKey] = {}
        self._lock = threading.RLock()
        logger.info(f"ShortTermMemoryStore 初始化, max_resident={max_resident_topics}")

    # ========== 辅助（锁内调用） ==========

    @staticmethod
    def _scope(workspace: WorkspaceIdentity) -> tuple[str, str]:
        return workspace.owner_user_id, workspace.workspace_id

    @staticmethod
    def _key(
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> WorkspaceTopicKey:
        identity_scope = require_identity_scope(identity_scope)
        return WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)

    def _require_buffer_locked(self, key: WorkspaceTopicKey) -> SemanticBuffer:
        """返回必须存在的话题 buffer；写命令不得静默忽略缺失 topic。"""
        buf = self._port.get(key)
        if buf is None:
            raise KeyError(f"topic '{key.topic_id}' does not exist in requested Workspace")
        return buf

    def _evict_locked(self, key: WorkspaceTopicKey) -> None:
        """从 Topic pool 移除 buffer，并修正 last-active 索引（锁内调用）。"""
        buf = self._port.pop(key)
        if buf is not None:
            scope = self._scope(buf.workspace_identity)
            if self._last_active_topic_keys.get(scope) == key:
                self._last_active_topic_keys.pop(scope, None)

    # ========== 最后活跃话题记录 ==========

    def get_last_active_topic(
        self,
        identity_scope: IdentityScope,
    ) -> Optional[str]:
        """返回当前 Workspace 最后活跃的 topic ID。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            key = self._last_active_topic_keys.get(self._scope(identity_scope.workspace_identity))
            return key.topic_id if key is not None else None

    def set_last_active_topic(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> None:
        key = self._key(identity_scope, topic_id)
        with self._lock:
            if self._port.get(key) is None:
                raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
            self._last_active_topic_keys[self._scope(identity_scope.workspace_identity)] = key

    # ========== CRUD ==========

    def get_topic_data(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> Optional[TopicData]:
        """Return an immutable topic read view without exposing SemanticBuffer."""
        key = self._key(identity_scope, topic_id)
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return None
            if touch:
                buf.last_accessed_at = datetime.now().timestamp()
                self._last_active_topic_keys[self._scope(identity_scope.workspace_identity)] = key
            return self._to_topic_data(buf)

    def get_topic_data_by_key(
        self,
        key: WorkspaceTopicKey,
        *,
        touch: bool = True,
    ) -> Optional[TopicData]:
        """由持有已验证复合键的内部协调器读取 Topic。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return None
            if touch:
                buf.last_accessed_at = datetime.now().timestamp()
                self._last_active_topic_keys[self._scope(buf.workspace_identity)] = key
            return self._to_topic_data(buf)

    def list_topic_data(
        self,
        identity_scope: IdentityScope,
        *,
        include_empty: bool = True,
    ) -> List[TopicData]:
        """Return immutable read views for active topics."""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            buffers = self._port.list_by_workspace(identity_scope.workspace_identity)
            if not include_empty:
                buffers = [buf for buf in buffers if buf.has_content]
            return [self._to_topic_data(buf) for buf in buffers]

    def topic_exists(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> bool:
        return self.get_topic_data(identity_scope, topic_id, touch=touch) is not None

    def has_blocks(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        data = self.get_topic_data(identity_scope, topic_id, touch=False)
        return bool(data and data.blocks)

    def create_buffer(
        self,
        identity_scope: IdentityScope,
        topic_title: str = "新建话题",
        topic_summary: str = "",
    ) -> TopicData:
        """创建话题并返回其冻结只读快照，不向调用方泄漏可变 SemanticBuffer。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            buf = SemanticBuffer(
                workspace_identity=identity_scope.workspace_identity,
                topic_title=topic_title,
                topic_summary=topic_summary,
            )
            self._port.put(buf.topic_key, buf)
            logger.debug(
                "创建话题段: topic_id=%s, owner=%s, workspace=%s",
                buf.topic_id,
                buf.workspace_identity.owner_user_id,
                buf.workspace_identity.workspace_id,
            )
            return self._to_topic_data(buf)

    def pop_buffer(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> Optional[TopicData]:
        """移除话题并返回移除前的冻结快照；不存在时返回 None。"""
        key = self._key(identity_scope, topic_id)
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return None
            snapshot = self._to_topic_data(buf)
            self._evict_locked(key)
            logger.info(f"移除话题段: topic_id={topic_id}")
            return snapshot

    def pop_buffer_by_key(self, key: WorkspaceTopicKey) -> Optional[TopicData]:
        """从显式生命周期入口驱逐 IDLE Topic；busy 时拒绝删除。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return None
            if buf.state is not BufferState.IDLE:
                raise TopicBusyError(
                    f"topic '{key.topic_id}' 正忙，无法执行显式驱逐"
                )
            snapshot = self._to_topic_data(buf)
            self._evict_locked(key)
            return snapshot

    # ========== 写操作（命名方法，禁止调用方直接写 buffer 字段）==========

    def add_block(self, key: WorkspaceTopicKey, block: LogicalBlock) -> None:
        with self._lock:
            buf = self._require_buffer_locked(key)
            buf.blocks.append(block)
            buf.total_tokens += block.total_tokens
            buf.last_update = datetime.now().timestamp()

    def clear_blocks(self, key: WorkspaceTopicKey) -> None:
        """清空 blocks 并重置 token 计数（settle 内部的瞬时步骤，非生命周期入口）。"""
        with self._lock:
            buf = self._require_buffer_locked(key)
            buf.blocks.clear()
            buf.total_tokens = 0
            buf.last_update = datetime.now().timestamp()

    def update_summary(
        self,
        key: WorkspaceTopicKey,
        summary: str,
    ) -> None:
        """只更新 state summary，不改变当前 blocks。"""
        with self._lock:
            buf = self._require_buffer_locked(key)
            buf.state_summary = summary
            buf.last_update = datetime.now().timestamp()

    def apply_compaction(
        self,
        key: WorkspaceTopicKey,
        summary: str,
        *,
        retain_count: int,
    ) -> int:
        """在持有 PROCESSING 时应用 compact 结果，返回被裁剪的 block 数。

        所有 compact 路径都必须保证至少保留一个最新 block；传入小于 1 的
        ``retain_count`` 在输入边界以具体异常拒绝，不静默提升。
        """
        if retain_count < 1:
            raise ValueError("retain_count must be >= 1")

        with self._lock:
            buf = self._require_buffer_locked(key)
            if buf.state is not BufferState.PROCESSING:
                raise TopicBusyError(
                    f"topic '{key.topic_id}' 未持有 PROCESSING 预约，不能应用 compact"
                )
            buf.state_summary = summary
            buf.last_update = datetime.now().timestamp()
            if len(buf.blocks) <= retain_count:
                return 0
            folded = len(buf.blocks) - retain_count
            buf.blocks = buf.blocks[-retain_count:]
            buf.total_tokens = sum(b.total_tokens for b in buf.blocks)
            return folded

    def update_title(self, key: WorkspaceTopicKey, title: str) -> None:
        """写入 topic_title（替代 buffer.topic_title = title）。"""
        with self._lock:
            buf = self._require_buffer_locked(key)
            buf.topic_title = title

    def update_metadata(self, key: WorkspaceTopicKey, state: Optional[BufferState] = None) -> None:
        with self._lock:
            buf = self._require_buffer_locked(key)
            if state is not None:
                buf.state = state
            buf.last_update = datetime.now().timestamp()

    def update_model_used(self, key: WorkspaceTopicKey, model_used: str) -> None:
        """写入最近一次 run 使用的模型展示名。"""
        with self._lock:
            buf = self._require_buffer_locked(key)
            buf.model_used = model_used

    # ========== P5：单写者预约 ==========

    def reserve_processing(self, key: WorkspaceTopicKey) -> bool:
        """IDLE -> PROCESSING；仅 IDLE Topic 可预约，busy 或缺失返回 False。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is None or buf.state is not BufferState.IDLE:
                return False
            buf.state = BufferState.PROCESSING
            return True

    def release_processing(self, key: WorkspaceTopicKey) -> None:
        """PROCESSING -> IDLE；幂等，仅在当前处于 PROCESSING 时释放。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is not None and buf.state is BufferState.PROCESSING:
                buf.state = BufferState.IDLE

    def reserve_flushing(self, key: WorkspaceTopicKey) -> bool:
        """IDLE -> FLUSHING；仅 IDLE Topic 可进入 manual settle 窗口。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is None or buf.state is not BufferState.IDLE:
                return False
            buf.state = BufferState.FLUSHING
            return True

    def commit_flushing(self, key: WorkspaceTopicKey) -> bool:
        """FLUSHING -> 驱逐；返回是否实际结束了 Topic 生命周期。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is None or buf.state is not BufferState.FLUSHING:
                return False
            self._evict_locked(key)
            return True

    def abort_flushing(self, key: WorkspaceTopicKey) -> None:
        """FLUSHING -> IDLE；admission 失败时恢复可写状态，保留 Topic 内容。"""
        with self._lock:
            buf = self._port.get(key)
            if buf is not None and buf.state is BufferState.FLUSHING:
                buf.state = BufferState.IDLE

    # ========== P5：原子 apply 与 binding ==========

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
        """在单个临界区内原子提交本轮 block、首次 bindings 与 metadata。

        前置：目标 Topic 必须已持有 ``PROCESSING`` 预约。binding 只在本轮
        Interaction 显式携带且尚未绑定的 ``asset_id`` 上首次建立，重复使用同一
        资产只幂等命中既有关系，不覆盖首次 Interaction 或首次绑定时间。
        """
        key = WorkspaceTopicKey.from_identity_scope(
            require_identity_scope(identity_scope),
            topic_id,
        )
        normalized_refs = self._normalize_asset_id_and_refs(asset_id_and_refs)
        if normalized_refs and not interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")

        with self._lock:
            buf = self._require_buffer_locked(key)
            if buf.state is not BufferState.PROCESSING:
                raise TopicBusyError(
                    f"topic '{topic_id}' 未持有 PROCESSING 预约，不能原子 apply interaction"
                )

            buf.blocks.append(block)
            buf.total_tokens += block.total_tokens

            now = datetime.now().timestamp()
            existing_ids = {binding.asset_id for binding in buf.bindings}
            for asset_id, asset_ref in normalized_refs:
                if asset_id in existing_ids:
                    continue
                buf.bindings.append(
                    TopicAssetBinding(
                        asset_id=asset_id,
                        asset_ref=asset_ref,
                        first_bound_interaction_id=interaction_id,
                        bound_at=datetime.fromtimestamp(now),
                    )
                )
                existing_ids.add(asset_id)

            if model_used:
                buf.model_used = model_used
            buf.last_update = now
            return self._to_topic_data(buf)

    def list_asset_bindings(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> tuple[TopicAssetBinding, ...]:
        """返回 Topic 当前冻结的资产使用关系；不存在 Topic 时返回空元组。"""
        key = self._key(identity_scope, topic_id)
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return ()
            return tuple(buf.bindings)

    def freeze_and_evict(self, key: WorkspaceTopicKey) -> Optional[TopicData]:
        """automatic settle 的原子 freeze-and-evict：仅接受 IDLE Topic。

        在一个临界区内冻结 blocks/state summary/binding refs、移除 buffer 并修正
        last-active 索引；缺失时返回 None，检测到 PROCESSING/FLUSHING 时显式抛出
        ``TopicBusyError``，避免上层把状态冲突误判为正常 generation skip。
        """
        with self._lock:
            buf = self._port.get(key)
            if buf is None:
                return None
            if buf.state is not BufferState.IDLE:
                raise TopicBusyError(
                    f"topic '{key.topic_id}' 正忙，无法执行 automatic settle"
                )
            snapshot = self._to_topic_data(buf)
            self._evict_locked(key)
            return snapshot

    def freeze_for_manual_settle(self, key: WorkspaceTopicKey) -> Optional[TopicData]:
        """manual settle 的 FLUSHING prepare：IDLE -> FLUSHING 并冻结快照。

        不清除 blocks、不驱逐 buffer；admission 成功后由 ``commit_flushing`` 驱逐，
        失败由 ``abort_flushing`` 恢复 IDLE。
        """
        with self._lock:
            buf = self._port.get(key)
            if buf is None or buf.state is not BufferState.IDLE:
                return None
            buf.state = BufferState.FLUSHING
            return self._to_topic_data(buf)

    # ========== LRU ==========

    def get_lru_topic(self, identity_scope: IdentityScope) -> Optional[str]:
        """返回访问时间最久远的 IDLE 话题 topic_id，跳过 busy 候选，无候选返回 None。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            bufs = self._port.list_by_workspace(identity_scope.workspace_identity)
            idle = [buf for buf in bufs if buf.state is BufferState.IDLE]
            if not idle:
                return None
            return min(idle, key=lambda b: b.last_accessed_at).topic_id

    def needs_eviction(self, identity_scope: IdentityScope) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            return self._port.count(identity_scope.workspace_identity) >= self.max_resident_topics

    def get_active_topic_buffer_count(self, identity_scope: IdentityScope) -> int:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            return self._port.count(identity_scope.workspace_identity)

    # ========== info ==========

    def get_buffer_info(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> Dict[str, Any]:
        with self._lock:
            buf = self._port.get(self._key(identity_scope, topic_id))
            if buf:
                return {
                    "exists": True,
                    "topic_id": buf.topic_id,
                    "block_count": len(buf.blocks),
                    "total_tokens": buf.total_tokens,
                    "has_content": buf.has_content,
                    "state": buf.state.value if hasattr(buf.state, "value") else buf.state,
                }
            return {"exists": False}

    def _to_topic_data(self, buf: SemanticBuffer) -> TopicData:
        return TopicData(
            topic_id=buf.topic_id,
            workspace_identity=buf.workspace_identity,
            current_agent_id=buf.current_agent_id,
            topic_title=buf.topic_title,
            topic_summary=buf.topic_summary,
            state_summary=buf.state_summary,
            blocks=tuple(buf.blocks),
            bindings=tuple(buf.bindings),
            state=buf.state,
            last_update=buf.last_update,
            last_accessed_at=buf.last_accessed_at,
            total_tokens=buf.total_tokens,
            model_used=buf.model_used,
        )

    def list_all_topic_data_for_maintenance(self) -> List[TopicData]:
        """供进程级 idle/shutdown 协调器遍历，不作为用户授权入口。"""
        with self._lock:
            return [self._to_topic_data(buf) for buf in self._port.list_all()]

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()

    @staticmethod
    def _normalize_asset_id_and_refs(
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...],
    ) -> list[tuple[str, WorkspaceAssetRef]]:
        """校验 asset_id + asset_ref 关系输入，拒绝空白 ID 与非 WorkspaceAssetRef。"""
        result: list[tuple[str, WorkspaceAssetRef]] = []
        for item in asset_id_and_refs:
            asset_id, asset_ref = item
            if not isinstance(asset_id, str) or not asset_id.strip():
                raise ValueError("asset_id 不能为空")
            if not isinstance(asset_ref, WorkspaceAssetRef):
                raise TypeError("asset_ref 必须是 WorkspaceAssetRef")
            result.append((asset_id.strip(), asset_ref))
        return result


# ============ MidTermMemoryStore ============

class MidTermMemoryStore:
    """
    中期记忆存储（向量库）

    持有 primary Port（向量库）和 optional secondary Ports（图库等）。
    写入时同步到所有后端；查询仅走 primary。
    """

    def __init__(
        self,
        primary: MidTermStoragePort,
        secondary: Optional[List[MidTermStoragePort]] = None,
    ) -> None:
        self._primary = primary
        self._secondary: List[MidTermStoragePort] = secondary or []

    async def upsert(self, memory: MemoryAtom) -> None:
        """仅持久化已通过 v2 领域校验的 canonical Memory。"""
        await self._primary.upsert(memory)
        for s in self._secondary:
            await s.upsert(memory)

    async def get(
        self,
        scope: IdentityScope,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        return await self._primary.get(require_identity_scope(scope), memory_id)

    async def get_by_alias(
        self,
        scope: IdentityScope,
        alias: str,
    ) -> Optional[MemoryAtom]:
        return await self._primary.get_by_alias(require_identity_scope(scope), alias)

    async def get_for_mutation(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> Optional[MemoryAtom]:
        identity_scope = require_identity_scope(identity_scope)
        return await self._primary.get_for_mutation(identity_scope, memory_id)

    async def get_by_key(self, key: WorkspaceMemoryKey) -> Optional[MemoryAtom]:
        return await self._primary.get_by_key(key)

    async def update_access_info(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> None:
        identity_scope = require_identity_scope(identity_scope)
        await self._primary.update_access_info(identity_scope, memory_id)

    async def delete(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        result = await self._primary.delete(identity_scope, memory_id)
        for s in self._secondary:
            await s.delete(identity_scope, memory_id)
        return result

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        result = await self._primary.delete_by_key(key)
        for secondary in self._secondary:
            await secondary.delete_by_key(key)
        return result

    async def batch_delete(
        self,
        identity_scope: IdentityScope,
        ids: List[UUID],
    ) -> int:
        identity_scope = require_identity_scope(identity_scope)
        count = await self._primary.batch_delete(identity_scope, ids)
        for s in self._secondary:
            await s.batch_delete(identity_scope, ids)
        return count

    async def search(
        self,
        scope: IdentityScope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        scope = require_identity_scope(scope)
        return await self._primary.search(
            scope,
            query,
            top_k,
            filters,
            mode,
            score_threshold,
        )

    async def scroll(
        self,
        scope: IdentityScope,
        filters=None,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        return await self._primary.scroll(
            require_identity_scope(scope),
            filters,
            limit,
        )

    async def count(self, scope: IdentityScope, filters=None) -> int:
        return await self._primary.count(require_identity_scope(scope), filters)

    async def list_all_for_maintenance(self, limit: int = 10000) -> List[MemoryAtom]:
        """供进程级生命周期维护遍历，不作为用户授权入口。"""
        return await self._primary.list_all_for_maintenance(limit)

    async def check_health(self) -> StorageHealthComponent:
        primary_health = await self._primary.check_health()
        if not primary_health.healthy:
            return primary_health

        for index, secondary in enumerate(self._secondary):
            secondary_health = await secondary.check_health()
            if not secondary_health.healthy and secondary_health.required:
                return StorageHealthComponent(
                    name=f"mid_term.secondary.{index}",
                    healthy=False,
                    required=True,
                    detail=secondary_health.detail,
                )

        return primary_health


# ============ LongTermMemoryStore ============

class LongTermMemoryStore:
    """
    长期记忆存储（冷存储）

    持有 LongTermStoragePort 实现，只负责读写冷存储。
    archive / revive 跨层操作由 MemoryLibrary 编排。
    """

    def __init__(self, port: LongTermStoragePort) -> None:
        self._port = port

    async def persist(self, memory: MemoryAtom) -> None:
        await self._port.persist(memory)

    async def load(self, key: WorkspaceMemoryKey) -> MemoryAtom:
        return await self._port.load(key)

    async def remove(self, key: WorkspaceMemoryKey) -> None:
        await self._port.remove(key)

    async def is_archived(self, key: WorkspaceMemoryKey) -> bool:
        return await self._port.is_archived(key)

    async def query(
        self,
        limit: int = 100,
        vitality_threshold: Optional[float] = None,
    ) -> List[ArchiveRecord]:
        return await self._port.query(limit=limit, vitality_threshold=vitality_threshold)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


# ============ ArtifactStore ============

class ArtifactStore:
    """Artifact 附属资产仓库（书库隐喻的附属档案室）。"""

    def __init__(self, port: ArtifactStoragePort) -> None:
        self._port = port

    async def put(self, artifact) -> ArtifactRef:
        return await self._port.put(artifact)

    async def get(self, identity_scope: IdentityScope, ref_or_id) -> dict:
        identity_scope = require_identity_scope(identity_scope)
        return await self._port.get(identity_scope, ref_or_id)

    async def exists(
        self,
        identity_scope: IdentityScope,
        artifact_id: str,
    ) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        return await self._port.exists(identity_scope, artifact_id)

    async def list_by_memory(
        self,
        identity_scope: IdentityScope,
        memory_id: str,
        artifact_type=None,
    ) -> list:
        identity_scope = require_identity_scope(identity_scope)
        return await self._port.list_by_memory(identity_scope, memory_id, artifact_type)

    async def verify(
        self,
        identity_scope: IdentityScope,
        ref,
    ) -> ArtifactIntegrityResult:
        identity_scope = require_identity_scope(identity_scope)
        return await self._port.verify(identity_scope, ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = [
    "ShortTermMemoryStore",
    "MidTermMemoryStore",
    "LongTermMemoryStore",
    "ArtifactStore",
]
