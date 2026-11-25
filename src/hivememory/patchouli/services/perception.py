"""Patchouli 感知使魔。

感知业务的唯一编排者：持有 Engine（纯算法，含 RelayController）、Store（内容）
与 WorkingSet（驻留与 lease），负责话题路由、Interaction 原子写入、话题折叠、
统一 settle 时序、LRU / idle / shutdown 维护，并把领域结果投影为公开报告。

结算分成两件事：沉淀把话题新的交互交给生成，话题留在话题池；退役让话题离开话题池，
退役前先沉淀还没有沉淀的交互。空闲与关闭只沉淀；LRU 与手动结算退役。用户会断断续续地
回到同一件事，空闲或重启时退役会把连续的工作切成几个话题。从没接收过交互的空话题没有
可以接续的工作，空闲与关闭时直接退役。

折叠是话题自身的维护，不属于交互应用：应用在写入 block 后即完成，是否需要
折叠由 ``needs_compact`` 判断，``compact_topic`` 作为独立用例执行。话题只经
core 话题模型的写函数写入：写入一轮只追加交互记录，折叠只追加折叠记录并
推进工作视图，原文在话题结束前一直保留。

占用权统一由 WorkingSet 的 lease 表达：交互、折叠与结算的写路径都是
``acquire → try/finally release``；Relay 摘要生成在 lease 持有期间进行，
其他调用无法取得同一话题的占用权，因此无需旧的记录状态补偿校验。只有本话题的
写入记录不取 lease：它与交互、折叠和沉淀进度互不相交。全部写回都经短期存储的
``update`` 套用在最新快照上，等待摘要或接纳期间记下的写入不会被覆盖。
统一 settle 时序（``_settle_topic``）为所有来源共用；是否退役由调用的用例决定，
``reason`` 只作为材料的 provenance 标签，不驱动分支。

话题池在正常关闭时由 runtime 写成快照、下次启动时读回（``restore_topics``）：关闭时
只沉淀、不退役，话题带着交互记录、折叠记录、沉淀进度与写入记录跨过重启。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    TopicData,
    WorkspaceAssetRef,
    WorkspaceIdentity,
    fold_topic,
    mark_topic_settled,
    merge_interaction_into_topic,
    record_topic_writes,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import TopicEvictionResult, TopicSettleResult
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet

if TYPE_CHECKING:
    from hivememory.config.patchouli import MemoryPerceptionConfig
    from hivememory.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


class PerceptionFamiliar:
    """感知业务门面：Engine + Store + WorkingSet 的编排者。

    ``engine=None`` 表示感知关闭（原 ``NullPerceptionLayer`` 语义）：摄入
    路径无任何存储副作用；维护用例对空存储自然空转。

    话题的建立、结算与不经结算的删除各发一个 ``topic.*`` 事件；事件只是旁路
    观测，发布失败不影响话题的存储与结算。
    """

    def __init__(
        self,
        *,
        engine: MemoryPerceptionEngine | None,
        store: ShortTermMemoryStore,
        working_set: TopicWorkingSet,
        bus: PatchouliBus,
        config: MemoryPerceptionConfig,
        runtime_events: RuntimeEventPublisher | None = None,
    ) -> None:
        self._engine = engine
        self._store = store
        self._working_set = working_set
        self._bus = bus
        # 空闲时限在每次扫描时读取，配置对象被原地改写后下一次扫描即按新值判断。
        self._config = config
        self._events = (runtime_events or RuntimeEventPublisher(NullRuntimeEventSink())).scoped(
            subsystem="patchouli", component="perception"
        )

    # ========== Interaction 摄入 ==========

    async def apply_interaction(
        self,
        payload: InteractionPayload,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        asset_refs: tuple[WorkspaceAssetRef, ...] = (),
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
    ) -> str:
        """应用一份已接纳的交互载荷并完成话题路由（核心用例）。

        本轮 block 与首次 asset binding 一次写入话题即完成，不执行折叠。写入是
        这次应用中最后一个可能失败的步骤：应用要么失败且没有写入，要么写入并
        成功。提交队列只允许未成功结束的交互重新提交，因此同一轮不会被写入两次；
        在写入之后加入可能失败的步骤会破坏这一点。
        目标为 ``NEW_TOPIC`` 时以提交携带的标题与摘要建立话题，后续路由
        依赖候选话题的标题与摘要；缺省时使用默认标题。
        """
        if self._engine is None:  # 感知关闭：无任何存储副作用
            return target_topic_id

        logger.debug(
            "PerceptionFamiliar 应用交互载荷: interaction_id=%s target_topic_id=%s",
            interaction_id,
            target_topic_id,
        )
        await self._maybe_evict_lru(belong_to, target_topic_id)
        topic_id = self._ensure_topic(
            belong_to,
            target_topic_id,
            created_by="interaction",
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )
        block = self._engine.build_block(payload, from_actor)

        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法原子摄入 interaction，可稍后重试")
        try:
            self._update_leased_topic(
                belong_to,
                topic_id,
                lambda topic: merge_interaction_into_topic(
                    topic, block, asset_refs, interaction_id, payload.model_used
                ),
            )
        finally:
            self._working_set.release(lease)

        self._refresh_residency(belong_to, topic_id)
        return topic_id

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        belong_to: WorkspaceIdentity,
    ) -> str:
        """确保目标短期话题存在（必要时先执行 LRU 驱逐），返回真实 topic_id。"""
        if self._engine is None:  # 感知关闭：不创建话题
            self._refresh_residency(belong_to, target_topic_id)
            return target_topic_id
        await self._maybe_evict_lru(belong_to, target_topic_id)
        topic_id = self._ensure_topic(
            belong_to,
            target_topic_id,
            created_by="prepare",
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )
        self._refresh_residency(belong_to, topic_id)
        return topic_id

    # ========== 统一 settle 协议 ==========

    async def _settle_topic(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        reason: TriggerReason,
        retire: bool,
        raise_on_admission_failure: bool,
    ) -> tuple[bool, MemoryGenerationTask | None]:
        """统一 settle 时序：获取 lease → 冻结材料 → 锁外 admission → 记下沉淀进度或退役。

        所有 settle 来源（manual / idle / LRU / shutdown）共用，只在接纳之后是否删除话题
        上分开：沉淀（``retire=False``）把话题留在存储与驻留中，记下这次覆盖到的交互数，
        不刷新访问时间；退役删除话题并移除驻留。最近一次沉淀之后没有新交互时不生成：
        沉淀什么也不做，退役直接删除话题。admission 失败时话题与沉淀进度原样保留可重试，
        无需 abort——记录从未被改动。
        返回 (这次沉淀或退役是否完成, 已接纳 task 或 None)。
        """
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始结算")
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                self._working_set.remove(belong_to, topic_id)  # 清理陈旧驻留条目
                return False, None
            # 冻结材料：字段转换与 no-material 判断收口在数据模型内。
            task = TopicMaterializeTask.from_topic_data(
                topic, belong_to=belong_to, reason=reason, retires_topic=retire
            )
            if task is None:  # 最近一次沉淀之后没有新交互，不触发生成
                if not retire:
                    return False, None
                self._store.delete(belong_to, topic_id)
                self._working_set.remove(belong_to, topic_id)
                self._emit_topic_settled(
                    belong_to, topic_id, reason, generation_submitted=False, retired=True
                )
                return True, None

            # admission：lease 持有期间等待 Generation queue 接纳。
            try:
                generation_task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, task
                )
            except Exception:
                if raise_on_admission_failure:
                    raise
                logger.warning(
                    "settle admission 失败，Topic 保留: topic_id=%s, reason=%s",
                    topic_id,
                    reason.value,
                    exc_info=True,
                )
                return False, None

            if retire:
                self._store.delete(belong_to, topic_id)
                self._working_set.remove(belong_to, topic_id)
            else:
                # lease 持有期间交互记录不会变化，冻结的交互数就是这次沉淀覆盖到的位置。
                settled_count = len(task.interactions)
                self._update_leased_topic(
                    belong_to, topic_id, lambda topic: mark_topic_settled(topic, settled_count)
                )
            self._emit_topic_settled(
                belong_to, topic_id, reason, generation_submitted=True, retired=retire
            )
            return True, generation_task
        finally:
            self._working_set.release(lease)

    async def manual_settle_topic(
        self, belong_to: WorkspaceIdentity, topic_id: str
    ) -> TopicSettleResult:
        """手动结算指定话题：沉淀还没有沉淀的交互并退役；话题不存在时抛出
        :class:`ResourceNotFoundError`。

        队列明确拒绝接纳时抛出 :class:`TopicSettleAdmissionError`（话题内容保留
        可重试）；目标正忙是瞬态冲突，保持 :class:`TopicBusyError` 语义。
        """
        if self._store.get(belong_to, topic_id) is None:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
        try:
            completed, task = await self._settle_topic(
                belong_to,
                topic_id,
                reason=TriggerReason.MANUAL_SETTLE,
                retire=True,
                raise_on_admission_failure=True,
            )
        except (WorkQueueCapacityError, WorkQueueStoppedError) as exc:
            raise TopicSettleAdmissionError(
                f"结算材料接纳失败，话题内容已保留，可重试: {topic_id}"
            ) from exc
        if not completed:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
        return TopicSettleResult(
            topic_id=topic_id, generation_task_id=task.task_id if task else None
        )

    # ========== Compact（页面折叠） ==========

    def needs_compact(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """话题超过折叠阈值且有可折叠的旧前缀时为真；话题不存在或感知关闭时为假。"""
        topic = self._store.get(belong_to, topic_id)
        return topic is not None and bool(self._blocks_to_fold(topic))

    async def compact_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """折叠用例：取得话题的 lease 后，在超过阈值时折叠工作视图的旧前缀。

        话题已被结算或驱逐时直接结束；正被占用时抛出 :class:`TopicBusyError`
        留待重试。摘要生成期间持有 lease，同一话题的写入与结算不会交错；写回在
        摘要返回后一次完成，中途取消或失败都不改动话题。
        """
        if self._engine is None:  # 感知关闭：不做任何折叠
            return
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法折叠，可稍后重试")
        try:
            await self._compact_topic_if_needed(belong_to, topic_id)
        finally:
            self._working_set.release(lease)

    def _blocks_to_fold(self, topic: TopicData) -> list[LogicalBlock]:
        """工作视图超过折叠阈值时返回保留后缀之前的旧 blocks；未超过或无可折叠前缀时为空。"""
        engine = self._engine
        if engine is None or topic.is_empty or not engine.should_compact(topic.total_tokens):
            return []
        # 保留块数以 Engine 配置为单一事实源。
        return engine.select_blocks_to_fold(topic.blocks, engine.config.fold_retain_recent_blocks)

    async def _compact_topic_if_needed(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """token 溢出时执行 compact：生成折叠摘要并追加一条折叠记录。

        调用方必须已持有该话题的 lease：Relay 摘要生成期间其他调用无法取
        得同一话题的占用权，因此写回不再需要旧的 expected_state 补偿校验。
        被折叠的 blocks 仍留在交互记录中，结算时随全部记录冻结。
        """
        engine = self._engine
        if engine is None:  # 感知关闭：与 apply 一致，不做任何折叠
            return
        topic = self._store.get(belong_to, topic_id)
        if topic is None:
            return
        blocks_to_fold = self._blocks_to_fold(topic)
        if not blocks_to_fold:
            return
        logger.info("Token 溢出，触发 Page Folding: topic_id=%s", topic_id)
        summary = await engine.generate_fold_summary(blocks_to_fold, topic.fold_summary)
        # lease 保证等待摘要期间交互记录与折叠记录没变，折叠前缀照样成立。
        self._update_leased_topic(
            belong_to, topic_id, lambda latest: fold_topic(latest, blocks_to_fold, summary)
        )

    # ========== 本话题的写入 ==========

    async def record_writes(
        self,
        topic_id: str,
        *,
        belong_to: WorkspaceIdentity,
        written: frozenset[str],
        absorbed_into: Mapping[str, str],
    ) -> None:
        """把一次生成执行后要记下的写入与合并改指记到话题上；话题已经退役时不记。

        不取 lease：写入记录与交互、折叠和沉淀进度互不相交，持有 lease 的用例同样把写函数
        套用在最新快照上。退役的话题不会再沉淀，它的记录没有读者。
        """
        self._store.update(
            belong_to, topic_id, lambda topic: record_topic_writes(topic, written, absorbed_into)
        )

    # ========== LRU / 驱逐 ==========

    async def _maybe_evict_lru(self, belong_to: WorkspaceIdentity, target_topic_id: str) -> None:
        """需要创建新话题且池满时，让 LRU 话题退役：先沉淀它还没有沉淀的交互。

        已有话题无需驱逐；未知目标必须直接拒绝。容量与候选由 WorkingSet
        决定；候选正被占用时改选其他话题。
        """
        if target_topic_id != "NEW_TOPIC":
            if self._store.get(belong_to, target_topic_id) is None:
                raise ResourceNotFoundError(
                    details={"resource": "topic", "topic_id": target_topic_id}
                )
            return
        if not self._working_set.needs_eviction(belong_to):
            return
        attempted: set[str] = set()
        while self._working_set.needs_eviction(belong_to):
            lru_topic_id = self._working_set.select_lru_candidate(belong_to, exclude=attempted)
            if lru_topic_id is None:  # 池满但全部候选正被占用，无法安全驱逐
                raise TopicBusyError("LRU 驱逐无可占用候选，稍后重试")
            try:
                completed, _ = await self._settle_topic(
                    belong_to,
                    lru_topic_id,
                    reason=TriggerReason.LRU_EVICTION,
                    retire=True,
                    raise_on_admission_failure=True,
                )
            except TopicBusyError:  # 候选在选择后进入 busy，改选其他话题
                attempted.add(lru_topic_id)
                continue
            if not completed:  # 候选已被其他生命周期操作移除，重查容量后改选
                attempted.add(lru_topic_id)
                continue
            return  # admission 成功或无材料完成后 Topic 已删除，容量释放

    async def evict_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算；正被占用时报 ``removed=False``。"""
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            return TopicEvictionResult(topic_id=topic_id, removed=False)
        try:
            removed = self._store.delete(belong_to, topic_id)
            if removed:
                self._working_set.remove(belong_to, topic_id)
                self._topic_events(belong_to, topic_id).emit(RuntimeEventType.TOPIC_DELETED)
            return TopicEvictionResult(topic_id=topic_id, removed=removed)
        finally:
            self._working_set.release(lease)

    # ========== 维护与 shutdown ==========

    async def scan_idle_topics_once(self) -> list[str]:
        """沉淀空闲超时话题中有新交互的，让从没接收过交互的空话题退役；返回被沉淀或退役的
        话题。候选由 WorkingSet 提供。

        沉淀之后话题留在话题池、访问时间不变，没有新交互就不会再被沉淀，同一段空闲
        因此不会反复沉淀；话题接收新交互后重新计时。空话题是为写入预先建立、这一轮却
        没有交互的话题，留在池中只占位置，空闲时直接退役、不生成。
        """
        flushed: list[str] = []
        for belong_to, topic_id in self._working_set.list_idle_candidates(
            self._config.idle_timeout_seconds
        ):
            logger.debug("检测到空闲话题: topic_id=%s", topic_id)
            topic = self._store.get(belong_to, topic_id)
            # 读取与 _settle_topic 取得 lease 之间没有 await，判断时的空话题在 lease 内仍为空。
            retire = topic is not None and topic.is_empty
            try:
                completed, _ = await self._settle_topic(
                    belong_to,
                    topic_id,
                    reason=TriggerReason.IDLE_TIMEOUT,
                    retire=retire,
                    raise_on_admission_failure=False,
                )
            except TopicBusyError:  # snapshot 后进入 busy，留给后续维护轮次
                continue
            if completed:
                flushed.append(topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """服务关闭前对全部驻留话题做与空闲扫描相同的处理，话题随后写进话题池快照。

        有未沉淀交互的沉淀、不退役；从没接收过交互的空话题退役；接收过交互、已经全部沉淀
        的不处理。单个话题的 busy 或 admission 异常被隔离记录到 ``failed_topic_ids``，
        不阻止其余话题：它带着未沉淀的交互照常保存，下次启动后由空闲扫描沉淀。
        """
        candidates = self._working_set.list_shutdown_candidates()
        resident_block_count = 0
        for belong_to, topic_id in candidates:
            topic = self._store.get(belong_to, topic_id)
            if topic is not None:
                resident_block_count += topic.block_count

        settled: list[str] = []
        retired: list[str] = []
        failed: list[str] = []
        for belong_to, topic_id in candidates:
            topic = self._store.get(belong_to, topic_id)
            # 读取与 _settle_topic 取得 lease 之间没有 await，判断时的空话题在 lease 内仍为空。
            retire = topic is not None and topic.is_empty
            try:
                completed, _ = await self._settle_topic(
                    belong_to,
                    topic_id,
                    reason=TriggerReason.SHUTDOWN,
                    retire=retire,
                    raise_on_admission_failure=True,
                )
            except Exception:
                logger.exception("shutdown settle 失败: topic_id=%s", topic_id)
                failed.append(topic_id)
                continue
            if not completed:  # 已经全部沉淀，或已被其他生命周期操作移除
                continue
            (retired if retire else settled).append(topic_id)

        logger.info(
            "shutdown flush 完成: settled=%d, retired=%d, failed=%d, blocks=%d",
            len(settled),
            len(retired),
            len(failed),
            resident_block_count,
        )
        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled),
            retired_topic_ids=tuple(retired),
            resident_block_count=resident_block_count,
            failed_topic_ids=tuple(failed),
        )

    def restore_topics(self, topics: Sequence[TopicData]) -> list[TopicData]:
        """读回上次关闭时保存的话题，返回超出容量、没有读回的话题。

        话题逐个写入存储，按 ``last_update`` 从旧到新登记驻留：最近写入的话题最后登记，
        最久没有写入的是第一个 LRU 候选。驻留的访问时间不随快照保存，这只是关闭前访问顺序
        的近似（折叠会推进 ``last_update``，为写入确认话题只刷新驻留）；空闲计时从读回时重新
        开始。同一 Workspace 的话题超过驻留容量（配置在两次运行之间调小）时只读回最近的
        若干个。在接收交互之前调用，存储与驻留集合此时为空。
        """
        restored: list[TopicData] = []
        over_capacity: list[TopicData] = []
        per_workspace: Counter[WorkspaceIdentity] = Counter()
        for topic in sorted(topics, key=lambda topic: topic.last_update, reverse=True):
            if per_workspace[topic.workspace_identity] >= self._working_set.max_resident:
                over_capacity.append(topic)
                continue
            per_workspace[topic.workspace_identity] += 1
            restored.append(topic)
        for topic in reversed(restored):
            self._store.put(topic)
            self._working_set.touch(topic.workspace_identity, topic.topic_id)
        return over_capacity

    # ========== 内部实现 ==========

    def _ensure_topic(
        self,
        belong_to: WorkspaceIdentity,
        target_topic_id: str,
        *,
        created_by: Literal["interaction", "prepare"],
        topic_title: str | None = None,
        topic_summary: str | None = None,
    ) -> str:
        """确保目标话题存在并返回真实 topic_id；未知目标显式拒绝。

        ``created_by`` 只用于新建话题的事件：应用交互时建立，或为写入预先确定
        话题时建立。
        """
        if target_topic_id == "NEW_TOPIC":
            topic = self._store.create(
                belong_to,
                topic_title=topic_title or "新建话题",
                topic_summary=topic_summary or "",
            )
            # 创建即入池：避免 Store 有记录而 WorkingSet 不可见的孤儿窗口。
            self._working_set.touch(belong_to, topic.topic_id)
            self._topic_events(belong_to, topic.topic_id).emit(
                RuntimeEventType.TOPIC_CREATED, data={"created_by": created_by}
            )
            return topic.topic_id
        if self._store.get(belong_to, target_topic_id) is None:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": target_topic_id})
        return target_topic_id

    def _update_leased_topic(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        write: Callable[[TopicData], TopicData],
    ) -> None:
        """持有 lease 时把写函数套用在话题的最新快照上。

        话题只在持有 lease 时删除，调用方持有它期间话题一定存在；不存在说明占用被绕过。
        """
        if self._store.update(belong_to, topic_id, write) is None:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})

    def _topic_events(self, belong_to: WorkspaceIdentity, topic_id: str) -> RuntimeEventPublisher:
        return self._events.bind(topic_id=topic_id, workspace_id=belong_to.workspace_id)

    def _emit_topic_settled(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        reason: TriggerReason,
        *,
        generation_submitted: bool,
        retired: bool,
    ) -> None:
        """话题被沉淀或退役；``retired`` 表示话题是否随之离开话题池，没有新交互的退役不提交生成。"""
        self._topic_events(belong_to, topic_id).emit(
            RuntimeEventType.TOPIC_SETTLED,
            data={
                "reason": reason.value,
                "generation_submitted": generation_submitted,
                "retired": retired,
            },
        )

    def _refresh_residency(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """更新驻留记录；话题已不存在时不复活陈旧条目。

        retry 的失败窗口内话题可能已被 settle 删除，此时以 Store 为准跳过
        touch，避免 idle/LRU 扫描空转。
        """
        if self._store.get(belong_to, topic_id) is None:
            return
        self._working_set.touch(belong_to, topic_id)


__all__ = ["PerceptionFamiliar"]
