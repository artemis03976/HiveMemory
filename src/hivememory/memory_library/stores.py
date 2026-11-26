"""Storage facades for Patchouli memory layers.

短期存储只暴露持久化事实：话题上每种事件一个写入方法，写入即在话题的日志末尾追加一条记录。
话题的折叠、沉淀、退役与路由策略不属于本模块；占用权（lease）由
``TopicWorkingSet`` 管理，编排由 Perception Familiar 承担。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from hivememory.core.errors import MemoryAliasConflictError
from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    MemoryAtom,
    TopicData,
    WorkspaceAssetRef,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import ArtifactRef
from hivememory.core.models.memory_change import MemoryChangeEvent
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.engines.perception.models import TriggerReason
from hivememory.memory_library.models import (
    ArtifactIntegrityResult,
    LoadedTopicLogs,
    StorageHealthComponent,
)
from hivememory.memory_library.ports import (
    ArtifactStoragePort,
    LongTermStoragePort,
    MemoryChangePublisherPort,
    MidTermStoragePort,
    ShortTermStoragePort,
)
from hivememory.memory_library.topic_log import (
    TopicCreatedRecord,
    TopicFoldRecord,
    TopicGeneratedMemoriesRecord,
    TopicInteractionRecord,
    TopicLogRecord,
    TopicProgressEntryRecord,
    TopicRetiredRecord,
    TopicSettledRecord,
    apply_fold,
    apply_generated_memories,
    apply_interaction,
    apply_progress_entry,
    apply_settled,
    decode_topic_record,
    encode_topic_record,
    replay_topic_log,
    topic_from_created,
)

logger = logging.getLogger(__name__)


class ShortTermMemoryStore:
    """短期话题的存储：每个话题一条只追加的日志，内存中放池中话题的读取值。

    话题上发生的每件事各有一个写入方法，写入就是在日志末尾追加一条记录。写入方法不检查话题
    处在什么状态：谁能写、什么时候写由 ``TopicWorkingSet`` 的 lease 保证，同一话题的各次写入一个
    接一个。每次写入在锁内依次：读当前值 → 用套用函数算出新值（构造校验失败直接抛出，什么都
    不写）→ 经 Port 追加这一行 → 替换内存中的值。先落盘再改内存：追加失败时内存不变，错误
    照常传播。写入方法在话题不存在时什么都不写，返回 ``None``。

    读取只读内存，不碰磁盘；磁盘只在写入时追加、在启动时由 :meth:`load` 读回。读取原样返回
    存储的 frozen 实例，调用方无法经引用改写存储的事实。

    退役的话题离开话题池，读取方法看不到它；它的日志保留到最后一次沉淀完成后再删除，其间只有
    :meth:`append_settled`、:meth:`append_generated_memories`、:meth:`get_generated_memories` 与 :meth:`delete` 还能
    找到它：沉淀完成与本话题生成的记忆在话题退役之后照样记、照样读。

    不职责：驻留容量、访问时间与 lease 属于 ``TopicWorkingSet``；沉淀、折叠与退役的编排属于
    Perception Familiar。
    """

    def __init__(self, port: ShortTermStoragePort) -> None:
        self._port = port
        # 话题 ID 全局唯一，读取时再核对 Workspace 归属。
        self._topics: dict[str, TopicData] = {}
        self._retired: dict[str, TopicData] = {}
        # 保护“读当前值 → 追加 → 替换”的原子性；工作集与生命周期决策不在本模块。
        self._lock = threading.RLock()

    # ========== 读取 ==========

    def get(self, belong_to: WorkspaceIdentity, topic_id: str) -> TopicData | None:
        """读取池中话题的读取值；访问追踪在 WorkingSet，不在此处。"""
        with self._lock:
            return _owned(self._topics.get(topic_id), belong_to)

    def list_by_workspace(
        self,
        belong_to: WorkspaceIdentity,
        *,
        include_empty: bool = True,
    ) -> list[TopicData]:
        """列出 Workspace 内池中的话题。"""
        with self._lock:
            return [
                topic
                for topic in self._topics.values()
                if topic.workspace_identity == belong_to and (include_empty or topic.has_content)
            ]

    def get_generated_memories(
        self, belong_to: WorkspaceIdentity, topic_id: str
    ) -> frozenset[str] | None:
        """本话题生成的记忆；退役话题的日志删除之前同样交出，话题不存在时返回 ``None``。

        只交出本话题生成的记忆：退役的话题不回到话题池，其余内容仍只经池中话题的读取方法交出。
        """
        with self._lock:
            topic = _owned(self._topics.get(topic_id), belong_to)
            if topic is None:
                topic = _owned(self._retired.get(topic_id), belong_to)
            return None if topic is None else topic.topic_generated_memory_ids

    # ========== 写入 ==========

    def create(
        self,
        belong_to: WorkspaceIdentity,
        topic_title: str = "新建话题",
        topic_summary: str = "",
        *,
        topic_id: str | None = None,
    ) -> TopicData:
        """建立新话题，写下它的第一条记录，返回初始读取值。

        话题 ID 全局唯一：给出的 ID 已被任何 Workspace 的话题占用时以 ``ValueError`` 拒绝。
        """
        record = TopicCreatedRecord(
            at=_now(),
            workspace_identity=belong_to,
            topic_title=topic_title,
            topic_summary=topic_summary,
        )
        topic_id = topic_id or str(uuid4())
        with self._lock:
            if topic_id in self._topics or topic_id in self._retired:
                raise ValueError(f"topic '{topic_id}' already exists")
            topic = topic_from_created(topic_id, record)
            self._port.append(topic_id, encode_topic_record(record))
            self._topics[topic_id] = topic
            return topic

    def append_interaction(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        block: LogicalBlock,
        asset_refs: tuple[WorkspaceAssetRef, ...] = (),
        interaction_id: str | None = None,
        model_used: str | None = None,
    ) -> TopicData | None:
        """话题接收一轮交互：追加到交互记录，并以 ``asset_id`` 幂等绑定这一轮的附件。"""
        record = TopicInteractionRecord(
            at=_now(),
            block=block,
            asset_refs=asset_refs,
            interaction_id=interaction_id,
            model_used=model_used,
        )
        return self._append(belong_to, topic_id, record, apply_interaction)

    def append_fold(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        folded: Sequence[LogicalBlock],
        summary: str,
    ) -> TopicData | None:
        """把工作视图开头的 ``folded`` 折叠为 ``summary``；不是工作视图的前缀时构造校验失败。"""
        record = TopicFoldRecord(
            at=_now(),
            folded_block_ids=tuple(block.block_id for block in folded),
            summary=summary,
        )
        return self._append(belong_to, topic_id, record, apply_fold)

    def append_settled(
        self, belong_to: WorkspaceIdentity, topic_id: str, *, settled_until: int
    ) -> TopicData | None:
        """记下一次沉淀完成、覆盖到第几个交互；话题的 ``settled_until`` 只往前推进。

        退役话题的日志在最后一次沉淀完成之前还在，这次写入同样记到它的日志上。
        """
        record = TopicSettledRecord(at=_now(), interaction_count=settled_until)
        return self._append(belong_to, topic_id, record, apply_settled, include_retired=True)

    def append_generated_memories(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        generated: Collection[str],
        absorbed_into: Mapping[str, str],
    ) -> TopicData | None:
        """记下本话题新写下的记忆与合并的改指。

        退役话题的日志在最后一次沉淀完成之前还在，这次写入同样记到它的日志上：最后一次沉淀
        执行时要认得排在它前面的同话题沉淀写下的记忆。
        """
        record = TopicGeneratedMemoriesRecord(
            at=_now(), generated=frozenset(generated), absorbed_into=dict(absorbed_into)
        )
        return self._append(
            belong_to, topic_id, record, apply_generated_memories, include_retired=True
        )

    def append_progress_entry(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        text: str,
        covered: Sequence[LogicalBlock],
        covered_until: int,
    ) -> TopicData | None:
        """在进展记录末尾追加一条：``covered`` 是它覆盖的那几轮，``covered_until`` 是覆盖之后的位置。

        这一条不是从当前覆盖到的位置开始时构造校验失败。只记到池中的话题：退役的话题不再有人
        读它的进展，找不到时什么都不写。
        """
        last = covered[-1].turn
        record = TopicProgressEntryRecord(
            at=_now(),
            text=text,
            from_turn_id=covered[0].turn.turn_id,
            to_turn_id=last.turn_id,
            to_event_seq=last.turn_events[-1].sequence,
            covered_until=covered_until,
        )
        return self._append(belong_to, topic_id, record, apply_progress_entry)

    def retire(
        self, belong_to: WorkspaceIdentity, topic_id: str, *, reason: TriggerReason
    ) -> TopicData | None:
        """话题退役：写下退役记录，读取值离开话题池，日志保留到最后一次沉淀完成。"""
        record = TopicRetiredRecord(at=_now(), reason=reason)
        with self._lock:
            topic = _owned(self._topics.get(topic_id), belong_to)
            if topic is None:
                return None
            self._port.append(topic_id, encode_topic_record(record))
            del self._topics[topic_id]
            self._retired[topic_id] = topic
            return topic

    def delete(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """删除话题的日志与读取值，池中的与已退役的都可以；话题不存在时返回 ``False``。

        是否允许删除（占用检查）由调用方持有 lease 判断。
        """
        with self._lock:
            if (
                _owned(self._topics.get(topic_id), belong_to) is None
                and _owned(self._retired.get(topic_id), belong_to) is None
            ):
                return False
            self._port.delete(topic_id)
            self._topics.pop(topic_id, None)
            self._retired.pop(topic_id, None)
            return True

    def _append[R: TopicLogRecord](
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        record: R,
        apply: Callable[[TopicData, R], TopicData],
        *,
        include_retired: bool = False,
    ) -> TopicData | None:
        """在话题上套用 ``record``、追加到日志并替换读取值；话题不存在时什么都不写。

        ``include_retired`` 为真时，池中没有的话题再到日志还在的退役话题中找。
        """
        with self._lock:
            table = self._topics
            topic = _owned(table.get(topic_id), belong_to)
            if topic is None and include_retired:
                table = self._retired
                topic = _owned(table.get(topic_id), belong_to)
            if topic is None:
                return None
            updated = apply(topic, record)
            self._port.append(topic_id, encode_topic_record(record))
            table[topic_id] = updated
            return updated

    # ========== 启动时读回 ==========

    def load(self) -> LoadedTopicLogs:
        """读回全部话题日志并重放；启动时、接收交互之前调用一次。

        没有退役的话题进入话题池；已经退役、最后一次沉淀没有完成的话题留着日志，等调用方重新
        发起这次沉淀；已经退役、沉淀都已完成的（删除日志之前进程退出）直接删除日志。某一行
        读不回、或重放时构造校验失败的日志整个移到一旁（rejected），这个话题不读回。
        """
        topics: list[TopicData] = []
        retired: list[tuple[TopicData, TopicRetiredRecord]] = []
        rejected: list[str] = []
        for topic_id, lines in self._port.read_all().items():
            try:
                topic, retired_record = replay_topic_log(
                    topic_id, [decode_topic_record(line) for line in lines]
                )
            except ValueError:
                rejected_file = self._port.reject(topic_id)
                logger.error(
                    "话题日志无法读回: reason=topic_log_unreadable, topic_id=%s, rejected_file=%s",
                    topic_id,
                    rejected_file,
                    exc_info=True,
                )
                rejected.append(rejected_file)
                continue
            if retired_record is None:
                topics.append(topic)
            elif topic.has_unsettled_interactions:
                retired.append((topic, retired_record))
            else:
                self._port.delete(topic_id)
        with self._lock:
            self._topics.update((topic.topic_id, topic) for topic in topics)
            self._retired.update((topic.topic_id, topic) for topic, _ in retired)
        return LoadedTopicLogs(topics=topics, retired=retired, rejected_files=rejected)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


def _owned(topic: TopicData | None, belong_to: WorkspaceIdentity) -> TopicData | None:
    """话题属于 ``belong_to`` 时原样返回；另一个 Workspace 的同名话题按不存在处理。"""
    if topic is None or topic.workspace_identity != belong_to:
        return None
    return topic


def _now() -> float:
    return datetime.now().timestamp()


class MidTermMemoryStore:
    """中期记忆存储（向量库）。

    alias 唯一性不变量：同一 Workspace 的中期库内，一个 alias
    至多被一条 Memory 占用。``upsert`` 是全部完整写入（生成、手工编辑、
    Profile 管理、revive）的汇聚点，也是唯一能覆盖所有路径的检查点，因此
    在主后端写入前校验；归档即释放 alias，revive 撞名显式失败。
    ``patch_payload`` 白名单不含 ``index.alias``，该路径无需校验。

    生产 Runtime 注入变更发布端口；四个提交入口都在 finally 中内联等待
    失效通知，提交失败与删除未命中同样通知，不回滚已完成的后端写入。
    例外是只改 ``meta.lifecycle`` 的 patch：动态统计字段独立成
    ``MemoryLifecycleState`` 聚合，正是为了让它们的更新不使读取视图失效。
    """

    # 动态状态聚合的 dotted 前缀；patch 路径全部落在其下时不发布变更通知。
    _LIFECYCLE_PATCH_PREFIX = "meta.lifecycle."

    # 唯一性判定只需确认"除自身外是否还有其他占用者"：自身至多一条，
    # 因此取两条即可覆盖。
    _ALIAS_HOLDER_PROBE_LIMIT = 2

    def __init__(
        self,
        primary: MidTermStoragePort,
        secondary: list[MidTermStoragePort] | None = None,
        *,
        change_publisher: MemoryChangePublisherPort | None = None,
    ) -> None:
        self._primary = primary
        self._secondary: list[MidTermStoragePort] = secondary or []
        self._change_publisher = change_publisher

    async def upsert(self, memory: MemoryAtom, *, recompute_vectors: bool = True) -> None:
        """提交完整 canonical Memory；primary 写入后沿顺序同步 secondary。

        写入前校验 alias 唯一性，冲突时抛 ``MemoryAliasConflictError`` 且不
        产生任何写入。沿用串行写入假设：并发写入者之间"检查→
        写入"的竞态不在保证范围内。
        """
        try:
            await self.ensure_alias_available(memory)
            await self._primary.upsert(memory, recompute_vectors=recompute_vectors)
            for secondary in self._secondary:
                await secondary.upsert(memory, recompute_vectors=recompute_vectors)
        finally:
            # primary 失败或 secondary 部分提交后也不能继续信任读取投影。
            await self._publish_change(memory.workspace_identity, memory.id, "upsert")

    async def list_alias_holders(
        self,
        workspace_identity: WorkspaceIdentity,
        alias: str,
        *,
        limit: int = _ALIAS_HOLDER_PROBE_LIMIT,
    ) -> list[UUID]:
        """返回 Workspace 中期库内占用 ``alias`` 的 memory_id（primary 为准）。"""
        return await self._primary.list_alias_holders(workspace_identity, alias, limit=limit)

    async def ensure_alias_available(self, memory: MemoryAtom) -> None:
        """校验 ``memory`` 的 alias 未被同 Workspace 的其他 Memory 占用。

        无 alias 的原子不参与唯一性约束；自身已持有该 alias（如保留原 alias
        的内容更新）视为可用。内容提交路径可在写版本 Artifact 之前调用本方法
        提前失败，避免冲突时留下孤立记录；``upsert`` 仍会再次校验。
        """
        alias = memory.index.alias
        if not alias:
            return
        holders = await self.list_alias_holders(memory.workspace_identity, alias)
        others = [holder for holder in holders if holder != memory.id]
        if others:
            raise MemoryAliasConflictError(
                "alias 已被同一 Workspace 内的其他记忆占用",
                details={
                    "alias": alias,
                    "memory_id": str(memory.id),
                    "conflicting_memory_id": str(others[0]),
                    "reason": "alias_occupied",
                },
            )

    async def get(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return await self._primary.get(
            belong_to,
            memory_id,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_by_alias(
        self,
        belong_to: WorkspaceIdentity,
        alias: str,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> MemoryAtom | None:
        return await self._primary.get_by_alias(
            belong_to,
            alias,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return await self._primary.get_by_key(key)

    async def patch_payload(
        self,
        key: WorkspaceMemoryKey,
        patch: Mapping[str, Any],
    ) -> MemoryAtom | None:
        """受限局部更新：primary 提交并返回结果，同一 patch 沿顺序同步 secondary。

        任一存储失败按顺序直接传播，不回滚已成功的写入（存储按顺序提交，没有跨存储事务）。只改
        ``meta.lifecycle`` 动态状态的 patch 不发布变更通知；涉及其他路径（如
        ``meta.access_policy``）时与其他提交入口一样在 finally 中通知。
        """
        affects_read_view = any(not path.startswith(self._LIFECYCLE_PATCH_PREFIX) for path in patch)
        try:
            result = await self._primary.patch_payload(key, patch)
            for secondary in self._secondary:
                await secondary.patch_payload(key, patch)
            return result
        finally:
            if affects_read_view:
                await self._publish_change(key.workspace_identity, key.memory_id, "patch")

    async def delete(self, belong_to: WorkspaceIdentity, memory_id: UUID) -> bool:
        """删除 canonical；未命中同样失效可能残留的读取投影。"""
        try:
            result = await self._primary.delete(belong_to, memory_id)
            for secondary in self._secondary:
                await secondary.delete(belong_to, memory_id)
            return result
        finally:
            await self._publish_change(belong_to, memory_id, "delete")

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        """按复合归属键删除；通知不依赖后端是否命中。"""
        try:
            result = await self._primary.delete_by_key(key)
            for secondary in self._secondary:
                await secondary.delete_by_key(key)
            return result
        finally:
            await self._publish_change(key.workspace_identity, key.memory_id, "delete")

    async def _publish_change(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: UUID,
        operation: Literal["upsert", "patch", "delete"],
    ) -> None:
        """通过注入端口内联通知，等待失效完成后才结束提交调用。"""
        if self._change_publisher is None:
            return
        await self._change_publisher.publish_change(
            MemoryChangeEvent(belong_to=belong_to, memory_id=memory_id, operation=operation)
        )

    async def search(
        self,
        belong_to: WorkspaceIdentity,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ):
        return await self._primary.search(
            belong_to,
            query,
            top_k,
            filters=filters,
            mode=mode,
            score_threshold=score_threshold,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def scroll(
        self,
        belong_to: WorkspaceIdentity,
        filters=None,
        limit: int = 100,
        *,
        cursor: str | None = None,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> tuple[list[MemoryAtom], str | None]:
        return await self._primary.scroll(
            belong_to,
            filters,
            limit,
            cursor=cursor,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def count(
        self,
        belong_to: WorkspaceIdentity,
        filters=None,
        *,
        from_actor: ActorIdentity,
        enforce_actor_visibility: bool = True,
    ) -> int:
        return await self._primary.count(
            belong_to,
            filters,
            from_actor=from_actor,
            enforce_actor_visibility=enforce_actor_visibility,
        )

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return await self._primary.list_all_for_maintenance(limit)

    async def check_health(self) -> StorageHealthComponent:
        primary_health = await self._primary.check_health()
        if not primary_health.healthy:
            return primary_health
        for index, secondary in enumerate(self._secondary):
            health = await secondary.check_health()
            if not health.healthy and health.required:
                return StorageHealthComponent(
                    name=f"mid_term.secondary.{index}",
                    healthy=False,
                    required=True,
                    detail=health.detail,
                )
        return primary_health


class LongTermMemoryStore:
    """长期记忆存储（冷存储）。"""

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
        vitality_threshold: float | None = None,
    ) -> list[ArchiveRecord]:
        return await self._port.query(limit=limit, vitality_threshold=vitality_threshold)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


class ArtifactStore:
    """Artifact 附属资产仓库。"""

    def __init__(self, port: ArtifactStoragePort) -> None:
        self._port = port

    async def put(self, artifact) -> ArtifactRef:
        return await self._port.put(artifact)

    async def get(self, belong_to: WorkspaceIdentity, ref_or_id) -> dict:
        return await self._port.get(belong_to, ref_or_id)

    async def exists(self, belong_to: WorkspaceIdentity, artifact_id: str) -> bool:
        return await self._port.exists(belong_to, artifact_id)

    async def list_by_memory(
        self,
        belong_to: WorkspaceIdentity,
        memory_id: str,
        artifact_type=None,
    ) -> list:
        return await self._port.list_by_memory(
            belong_to,
            memory_id,
            artifact_type,
        )

    async def verify(self, belong_to: WorkspaceIdentity, ref) -> ArtifactIntegrityResult:
        return await self._port.verify(belong_to, ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = ["ShortTermMemoryStore", "MidTermMemoryStore", "LongTermMemoryStore", "ArtifactStore"]
