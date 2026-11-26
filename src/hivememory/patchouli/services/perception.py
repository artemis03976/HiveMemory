"""Patchouli 感知使魔。

感知业务的唯一编排者：持有 Engine（纯算法，含 RelayController）、Store（内容）
与 WorkingSet（驻留与 lease），负责话题路由、Interaction 原子写入、话题折叠、
统一 settle 时序、LRU / idle 维护，并把领域结果投影为公开报告。

结算分成两件事：沉淀把话题新的交互交给生成，话题留在话题池；退役让话题离开话题池，
退役前先沉淀还没有沉淀的交互。空闲只沉淀；LRU 与手动结算退役。用户会断断续续地
回到同一件事，空闲或重启时退役会把连续的工作切成几个话题。从没接收过交互的空话题没有
可以接续的工作，空闲时直接退役。

折叠是话题自身的维护，不属于交互应用：应用在写入 block 后即完成，是否需要
折叠由 ``needs_fold`` 判断，``fold_topic`` 作为独立用例执行。话题经短期存储的
写入方法写入，每次写入在话题的日志末尾追加一条记录：写入一轮只追加交互记录，折叠只追加
折叠记录并推进工作视图，原文在话题结束前一直保留。

进展记录同样是话题自身的维护：话题停顿时，停顿扫描（``scan_paused_topics_once``）为它排入
写一条，与折叠一样经交互提交队列调度；``write_progress_entry`` 以旧进展记录与还没整理的交互
原文生成一段总结，只往后追加。

占用权统一由 WorkingSet 的 lease 表达：交互、折叠与结算的写路径都是
``acquire → try/finally release``；Relay 摘要生成在 lease 持有期间进行，
其他调用无法取得同一话题的占用权，因此无需旧的记录状态补偿校验。本话题生成的记忆、
沉淀完成与进展记录不取 lease：它们与交互和折叠互不相交。存储的每次写入都在最新的读取值上套用，
等待摘要或接纳期间记下的生成的记忆不会被覆盖。
统一 settle 时序（``_settle_topic``）为所有来源共用；是否退役由调用的用例决定，
``reason`` 只作为材料的 provenance 标签，不驱动分支。

沉淀进度在生成完成时推进：接纳时只在 WorkingSet 记下在途，生成完成后经
``record_settled`` 在日志中记下进度。退役的话题日志保留到最后一次沉淀完成；话题随时落盘，
启动时由 runtime 重放日志读回（``restore_topics``），并重新发起退役话题没有完成的最后一次
沉淀（``resettle_retired_topics``）。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TopicSettleMaterial, TriggerReason
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import TopicDeleteResult, TopicSettleResult
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet

if TYPE_CHECKING:
    from hivememory.config.patchouli import MemoryPerceptionConfig
    from hivememory.memory_library.stores import ShortTermMemoryStore
    from hivememory.memory_library.topic_log import TopicRetiredRecord

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

    async def receive_interaction(
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
        await self._retire_lru_for_new_topic(belong_to, target_topic_id)
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
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法接收这一轮交互，可稍后重试")
        try:
            written = self._store.append_interaction(
                belong_to,
                topic_id,
                block=block,
                asset_refs=asset_refs,
                interaction_id=interaction_id,
                model_used=payload.model_used,
            )
            self._require_leased_topic(written, topic_id)
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
        """确保目标短期话题存在（建新话题时池满，先让最久没用的话题退役），返回真实 topic_id。"""
        if self._engine is None:  # 感知关闭：不创建话题
            self._refresh_residency(belong_to, target_topic_id)
            return target_topic_id
        await self._retire_lru_for_new_topic(belong_to, target_topic_id)
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
        """统一 settle 时序：获取 lease → 冻结材料 → 锁外 admission → 记下在途或退役。

        所有 settle 来源（manual / idle / LRU）共用，只在接纳之后是否退役上分开：沉淀
        （``retire=False``）把话题留在存储与驻留中，不刷新访问时间；退役让话题离开存储与驻留。
        有没有新交互以日志中的沉淀进度与在途沉淀两者中较大的为准，在途期间同一段不会再沉淀；
        没有新交互时不生成：沉淀什么也不做，退役直接让话题退役。

        接纳之后只在 WorkingSet 记下在途，生成完成时由 :meth:`record_settled` 在日志中记下
        进度；协调器没有建任务（生成上下文为空）时没有可等的完成，立即记下。admission 失败时
        话题与沉淀进度原样保留可重试，无需 abort——记录从未被改动。
        返回 (这次沉淀或退役是否完成, 已接纳 task 或 None)。
        """
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始沉淀")
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                self._working_set.remove(belong_to, topic_id)  # 清理陈旧驻留条目
                return False, None
            # 冻结材料：字段转换与 no-material 判断收口在数据模型内。
            task = TopicSettleMaterial.from_topic_data(
                topic,
                belong_to=belong_to,
                reason=reason,
                retires_topic=retire,
                settled_count=max(
                    topic.settled_interaction_count,
                    self._working_set.settling_count(belong_to, topic_id),
                ),
            )
            if task is None:  # 已沉淀与在途的沉淀覆盖了全部交互，不触发生成
                if not retire:
                    return False, None
                self._retire_topic(belong_to, topic_id, reason)
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
                    "沉淀材料没有被生成队列接纳，话题保留: topic_id=%s, reason=%s",
                    topic_id,
                    reason.value,
                    exc_info=True,
                )
                return False, None

            # lease 持有期间交互记录不会变化，冻结的交互数就是这次沉淀覆盖到的位置。
            settled_count = len(task.interactions)
            if generation_task is None:
                self._store.append_settled(belong_to, topic_id, interaction_count=settled_count)
            else:
                self._working_set.mark_settling(belong_to, topic_id, settled_count)
            if retire:
                self._retire_topic(belong_to, topic_id, reason)
            self._emit_topic_settled(
                belong_to, topic_id, reason, generation_submitted=True, retired=retire
            )
            return True, generation_task
        finally:
            self._working_set.release(lease)

    def _retire_topic(
        self, belong_to: WorkspaceIdentity, topic_id: str, reason: TriggerReason
    ) -> None:
        """话题离开存储与驻留；还有沉淀没有完成时保留日志，等最后一次沉淀完成再删除。

        沉淀都已完成（包括从没接收过交互的空话题）时直接删除日志。调用方持有 lease。
        """
        topic = self._store.get(belong_to, topic_id)
        if topic is not None and topic.has_unsettled_interactions:
            self._store.retire(belong_to, topic_id, reason=reason)
        else:
            self._store.delete(belong_to, topic_id)
        self._working_set.remove(belong_to, topic_id)

    async def settle_and_retire_topic(
        self, belong_to: WorkspaceIdentity, topic_id: str
    ) -> TopicSettleResult:
        """手动沉淀并退役指定话题：沉淀还没有沉淀的交互，然后退役；话题不存在时抛出
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
                f"沉淀材料接纳失败，话题内容已保留，可重试: {topic_id}"
            ) from exc
        if not completed:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
        return TopicSettleResult(
            topic_id=topic_id, generation_task_id=task.task_id if task else None
        )

    # ========== 折叠 ==========

    def needs_fold(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """话题超过折叠阈值且有可折叠的旧前缀时为真；话题不存在或感知关闭时为假。"""
        topic = self._store.get(belong_to, topic_id)
        return topic is not None and bool(self._blocks_to_fold(topic))

    async def fold_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """折叠用例：取得话题的 lease 后，在超过阈值时折叠工作视图的旧前缀。

        话题已退役或删除时直接结束；正被占用时抛出 :class:`TopicBusyError`
        留待重试。摘要生成期间持有 lease，同一话题的交互写入与沉淀不会交错；写回在
        摘要返回后一次完成，中途取消或失败都不改动话题。
        """
        if self._engine is None:  # 感知关闭：不做任何折叠
            return
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法折叠，可稍后重试")
        try:
            await self._fold_if_needed(belong_to, topic_id)
        finally:
            self._working_set.release(lease)

    def _blocks_to_fold(self, topic: TopicData) -> list[LogicalBlock]:
        """工作视图超过折叠阈值时返回保留后缀之前的旧 blocks；未超过或无可折叠前缀时为空。"""
        engine = self._engine
        if engine is None or topic.is_empty or not engine.should_fold(topic.total_tokens):
            return []
        # 保留块数以 Engine 配置为单一事实源。
        return engine.select_blocks_to_fold(topic.blocks, engine.config.fold_retain_recent_blocks)

    async def _fold_if_needed(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """工作视图超过折叠阈值时折叠：生成折叠摘要并追加一条折叠记录。

        调用方必须已持有该话题的 lease：折叠摘要生成期间其他调用无法取
        得同一话题的占用权，因此写回不再需要旧的 expected_state 补偿校验。
        被折叠的 blocks 仍留在交互记录中，沉淀时随全部记录冻结。
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
        logger.info("工作视图超过折叠阈值，折叠: topic_id=%s", topic_id)
        summary = await engine.generate_fold_summary(blocks_to_fold, topic.fold_summary)
        # lease 保证等待摘要期间交互记录与折叠记录没变，折叠前缀照样成立。
        written = self._store.append_fold(
            belong_to, topic_id, folded=blocks_to_fold, summary=summary
        )
        self._require_leased_topic(written, topic_id)

    # ========== 进展记录 ==========

    async def write_progress_entry(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """写一条进展记录：以旧进展记录与还没整理的交互原文生成一段总结，追加到话题的日志上。

        不取 lease：这一条只要求从当前覆盖到的位置开始，这个位置只有写进展记录会改，同一话题
        同时至多一个维护项，交互只往后追加，生成期间别的写入不会破坏它；存储的位置校验兜住
        违反这一点的错误。生成期间沉淀与 LRU 照常取得 lease；话题因此退役或被删除时这一条不记。
        生成完成才追加，中途取消或进程退出都什么也不写。

        生成失败时在 WorkingSet 记下这次要覆盖到的位置后照常抛出：同一位置不再重试，否则提供商
        故障时停顿扫描每一轮都会再排入一次。
        """
        engine = self._engine
        if engine is None or not engine.writes_progress_record:
            return
        topic = self._store.get(belong_to, topic_id)
        if topic is None:
            return
        start = topic.recorded_interaction_count
        covered = topic.interactions[start:]
        if not covered:
            return
        covered_until = len(topic.interactions)
        try:
            text = await engine.generate_progress_entry(
                topic.progress_record, covered, first_turn_number=start + 1
            )
        except Exception:
            self._working_set.mark_progress_entry_failed(belong_to, topic_id, covered_until)
            raise
        written = self._store.append_progress_entry(
            belong_to, topic_id, text=text, covered=covered, covered_until=covered_until
        )
        if written is None:
            logger.info(
                "话题在写进展记录期间离开了话题池，这一条不记: topic_id=%s, covered_until=%s",
                topic_id,
                covered_until,
            )
            return
        logger.info(
            "写下一条进展记录: topic_id=%s, covered_turns=%s, covered_until=%s, chars=%s",
            topic_id,
            len(covered),
            covered_until,
            len(text),
        )

    # ========== 本话题生成的记忆与沉淀完成 ==========

    async def record_generated_memories(
        self,
        topic_id: str,
        *,
        belong_to: WorkspaceIdentity,
        generated: frozenset[str],
        absorbed_into: Mapping[str, str],
    ) -> None:
        """把一次生成执行后本话题新生成的记忆与合并改指记到话题上。

        退役话题的日志删除之前照样记：话题退役时可能还有排在前面的同话题沉淀，随退役执行的
        最后一次沉淀要认得它们写下的记忆。话题已被删除、或日志已经删除时不记。不取 lease：
        本话题生成的记忆与交互、折叠和沉淀进度互不相交，存储在最新的读取值上套用。
        """
        self._store.append_generated_memories(
            belong_to, topic_id, generated=generated, absorbed_into=absorbed_into
        )

    async def get_generated_memories(
        self, topic_id: str, *, belong_to: WorkspaceIdentity
    ) -> frozenset[str] | None:
        """本话题生成的记忆，退役话题的日志删除之前同样交出；话题不存在时为 ``None``。"""
        return self._store.get_generated_memories(belong_to, topic_id)

    async def record_settled(
        self, topic_id: str, *, belong_to: WorkspaceIdentity, interaction_count: int
    ) -> None:
        """一次沉淀完成：在日志中记下它覆盖到的交互数，清除不超过它的在途。

        话题在池中时读取值随之更新。话题已经退役、日志还在时照样记下；这一次完成之后再没有
        未沉淀的交互，就是最后一次沉淀完成，随即删除日志。话题已被手动删除、或日志已经删除
        时不记。不取 lease：沉淀进度与交互和折叠互不相交。
        """
        self._working_set.clear_settling(belong_to, topic_id, up_to=interaction_count)
        retired = self._store.get(belong_to, topic_id) is None
        topic = self._store.append_settled(belong_to, topic_id, interaction_count=interaction_count)
        if topic is not None and retired and not topic.has_unsettled_interactions:
            self._store.delete(belong_to, topic_id)

    # ========== 退役与删除 ==========

    async def _retire_lru_for_new_topic(
        self, belong_to: WorkspaceIdentity, target_topic_id: str
    ) -> None:
        """需要创建新话题且池满时，让 LRU 话题逐个退役，直到腾出位置：退役前先沉淀它还没有
        沉淀的交互。

        通常只退役一个；驻留容量在两次运行之间调小时，读回的话题超出容量，这里一次退役到容量
        以内。已有话题无需驱逐；未知目标必须直接拒绝。容量与候选由 WorkingSet 决定；候选正被
        占用时改选其他话题。
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
            if lru_topic_id is None:  # 池满但全部候选正被占用，无法安全退役
                raise TopicBusyError("话题池已满且全部话题正被占用，稍后重试")
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

    async def delete_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> TopicDeleteResult:
        """删除话题：删除它的日志与内存中的状态，不沉淀；正被占用时报 ``removed=False``。

        已经被接纳的沉淀照常执行，完成时话题已经不在，不再记下进度。
        """
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            return TopicDeleteResult(topic_id=topic_id, removed=False)
        try:
            removed = self._store.delete(belong_to, topic_id)
            if removed:
                self._working_set.remove(belong_to, topic_id)
                self._topic_events(belong_to, topic_id).emit(RuntimeEventType.TOPIC_DELETED)
            return TopicDeleteResult(topic_id=topic_id, removed=removed)
        finally:
            self._working_set.release(lease)

    # ========== 维护与启动时读回 ==========

    async def scan_idle_topics_once(self) -> list[str]:
        """沉淀空闲超时话题中有新交互的，让从没接收过交互的空话题退役；返回被沉淀或退役的
        话题。候选由 WorkingSet 提供。

        沉淀之后话题留在话题池、访问时间不变，没有新交互就不会再被沉淀，同一段空闲
        因此不会反复沉淀；话题接收新交互后重新计时。空话题是为写入预先建立、这一轮却
        没有交互的话题，留在池中只占位置，空闲时直接退役、不生成。
        """
        settled: list[str] = []
        for belong_to, topic_id in self._working_set.list_inactive(
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
                settled.append(topic_id)
        return settled

    async def scan_paused_topics_once(
        self, schedule_progress_entry: Callable[[WorkspaceIdentity, str], Awaitable[bool]]
    ) -> list[str]:
        """为停顿超过阈值、有还没整理的交互的话题排入写一条进展记录；返回排入了的话题。

        停顿看驻留的访问时间，候选与空闲扫描相同、只是阈值更短，空闲时通常已经写过一条。写下
        一条之后访问时间不变，没有新交互就不会再写；上一次在同一位置失败、之后没有新交互的
        话题跳过。``schedule_progress_entry`` 把一项维护排进交互提交队列，话题已有未结束的维护项
        时返回假，下一轮再看。
        """
        engine = self._engine
        if engine is None or not engine.writes_progress_record:
            return []
        scheduled: list[str] = []
        for belong_to, topic_id in self._working_set.list_inactive(
            self._config.progress_record_pause_seconds
        ):
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                continue
            interaction_count = len(topic.interactions)
            if topic.recorded_interaction_count == interaction_count:
                continue
            if self._working_set.progress_entry_failed_until(belong_to, topic_id) >= (
                interaction_count
            ):
                continue
            if await schedule_progress_entry(belong_to, topic_id):
                scheduled.append(topic_id)
        return scheduled

    def restore_topics(self, topics: Sequence[TopicData]) -> None:
        """登记启动时从日志读回的话题的驻留；在接收交互之前调用，驻留集合此时为空。

        按 ``last_update`` 从旧到新登记：最近写入的话题最后登记，最久没有写入的是第一个 LRU
        候选。驻留的访问时间不写进日志，这只是上次运行访问顺序的近似；空闲计时从读回时开始。
        同一 Workspace 的话题超过驻留容量（配置在两次运行之间调小）时全部读回，下一次需要
        新话题时由 LRU 逐个退役，未沉淀的交互不丢。
        """
        for topic in sorted(topics, key=lambda topic: topic.last_update):
            self._working_set.touch(topic.workspace_identity, topic.topic_id)

    async def resettle_retired_topics(
        self, retired: Sequence[tuple[TopicData, TopicRetiredRecord]]
    ) -> int:
        """重新发起上次运行中已退役、最后一次沉淀没有完成的话题的沉淀，返回发起的数量。

        在生成队列启动之后调用。材料由日志组装，与运行时的沉淀一样冻结全部交互；话题仍是退役，
        完成时经 :meth:`record_settled` 删除日志。生成队列没有接纳的话题保留日志，下次启动时
        再发起。
        """
        resubmitted = 0
        for topic, retired_record in retired:
            belong_to = topic.workspace_identity
            task = TopicSettleMaterial.from_topic_data(
                topic,
                belong_to=belong_to,
                reason=retired_record.reason,
                retires_topic=True,
                settled_count=topic.settled_interaction_count,
            )
            assert task is not None  # 读回时只交来还有未沉淀交互的话题
            try:
                generation_task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, task
                )
            except (WorkQueueCapacityError, WorkQueueStoppedError):
                logger.warning(
                    "退役话题的最后一次沉淀没有被接纳，日志保留到下次启动: "
                    "reason=topic_resettle_rejected, topic_id=%s",
                    topic.topic_id,
                    exc_info=True,
                )
                continue
            if generation_task is None:
                await self.record_settled(
                    topic.topic_id, belong_to=belong_to, interaction_count=len(task.interactions)
                )
            resubmitted += 1
        return resubmitted

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

    @staticmethod
    def _require_leased_topic(written: TopicData | None, topic_id: str) -> None:
        """持有 lease 时的写入必须找到话题。

        话题只在持有 lease 时删除或退役，调用方持有它期间话题一定在池中；找不到说明占用被
        绕过。
        """
        if written is None:
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
