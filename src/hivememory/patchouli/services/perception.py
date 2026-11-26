"""Patchouli 感知使魔：维护话题池中的话题。

感知是 Patchouli 中接收交互、维护话题的部分。它维护的对象叫话题（topic，即短期记忆）：一段讨论
或一项工作接收到的全部交互。话题的内容在 ``ShortTermMemoryStore``（每个话题一条只追加的日志，
随时落盘），进程内的驻留记录与占用在 ``TopicWorkingSet``，算法在 ``MemoryPerceptionEngine``；
本服务编排它们，不另存状态。

话题离开话题池有两种方式：退役，先沉淀还没有沉淀的交互，日志保留到最后一次沉淀完成；删除，
不沉淀，日志立即删除。沉淀是把话题的交互交给记忆生成，话题留在话题池。空闲时只沉淀、不退役：
用户会断断续续地回到同一件事，空闲时退役会把连续的工作切成几个话题。

话题的位置
==========

每个位置都是“处理到第几轮”，即覆盖到的交互数。日志里有四个，是 ``TopicData`` 的属性：
``interaction_count``、``folded_until``、``recorded_until`` 与 ``settled_until``；进程内有两个，
在驻留记录 ``ResidentTopic`` 上：``settling_until``（在途沉淀）与 ``progress_failed_until``
（写进展记录失败）。它们彼此独立推进，都不切分材料：沉淀总是冻结全部交互与折叠记录。

触发表
======

每个触发写四项：看什么、做什么、推进哪个位置、失败与关闭时怎样。

接收一轮交互（``receive_interaction``，交互提交队列调用）
    看什么：目标是不是新话题；新话题时池是否已满。
    做什么：池满时先让最久没用的话题退役（见“池满时为新话题腾位置”），建立新话题；取 lease
        追加这一轮；刷新访问时间。提交队列在应用之后看工作视图是否超过折叠阈值，超过时排一项
        折叠。
    推进：``interaction_count``。
    失败：话题正被占用时以 ``TopicBusyError`` 失败，提交队列稍后重试（有次数上限）；目标话题
        已不在池中时以 ``ResourceNotFoundError`` 失败，提交方可以重新路由后再交。追加这一轮是
        最后一个可能失败的步骤，失败时这一轮没有写入；已经建立的新话题没有交互，由空闲扫描退役。

折叠（``fold_topic``，交互提交队列的维护项）
    看什么：工作视图的 token 是否超过折叠阈值，轮数是否多于保留的最近几轮。
    做什么：取 lease，把工作视图除最近几轮外折叠为折叠摘要，追加一条折叠记录；交互记录不变。
    推进：``folded_until``。
    失败：摘要模型超时或提供商失败时以规则摘要接上；其他错误使这一项失败、不折叠，下一次接收
        交互时重新判断。话题正被占用时提交队列重试。关闭时进行中的被取消，什么也不写。

停顿（``scan_paused_topics_once`` 排入维护项，``write_progress_entry`` 执行）
    看什么：访问时间超过 ``progress_record_pause_seconds`` 且没有被占用；``recorded_until`` 之后
        有交互；``progress_failed_until`` 小于 ``interaction_count``。没有写进展记录的模型时不写。
    做什么：排一项维护；执行时不取 lease，以旧进展记录与 ``recorded_until`` 之后的原文写一条，
        写完才追加。
    推进：``recorded_until``。
    失败：生成失败时记下 ``progress_failed_until``，有新交互之前不再写；生成期间话题退役或删除
        时这一条不记。话题已有未结束的维护项时这一轮不排。关闭时进行中的被取消，什么也不写。

空闲（``scan_idle_topics_once``）
    看什么：访问时间超过 ``idle_timeout_seconds`` 且没有被占用；``settled_until`` 与
        ``settling_until`` 中较远的那个之后有没有交互。
    做什么：有新交互：取 lease，冻结全部交互与折叠记录交给生成队列，话题留在池中，访问时间不变。
        从没接收过交互：退役，日志直接删除。其余什么也不做。
    推进：``settling_until``；生成队列没有为它建任务（材料里没有可生成的轮次）时直接推进
        ``settled_until``。
    失败：接纳失败（任何异常）时只记日志，话题原样保留，下一轮再看；正被占用时这一轮跳过。

池满时为新话题腾位置（``_retire_lru_for_new_topic``）、手动沉淀并退役（``settle_and_retire_topic``）
    看什么：``settled_until`` 与 ``settling_until`` 中较远的那个之后有没有交互；池满时还看哪个
        话题最久没用、没有被占用。
    做什么：取 lease；有新交互先沉淀，然后退役：写退役记录，离开话题池，日志保留到最后一次沉淀
        完成；已经没有未沉淀的交互时直接删除日志。池满时从最久没用的话题开始，一直退役到池有
        空位。
    推进：话题退役，进程内的两个位置随驻留记录一起删除；生成队列没有为它建任务时先推进
        ``settled_until``，日志随即删除。
    失败：接纳失败时话题原样保留、错误交给调用方：池满时这一轮交互或话题准备失败；手动时队列
        已满或已停止报为 ``TopicSettleAdmissionError``。全部候选正被占用、或手动的目标正被占用时
        以 ``TopicBusyError`` 失败。

沉淀到达终态（``record_settled``，生成控制器经本地路由调用）
    看什么：话题在池中还是已经退役。
    做什么：记下这次沉淀覆盖到第几轮；话题已经退役、这一次之后再没有未沉淀的交互时删除日志。
    推进：``settled_until``（只往前），清除不超过它的 ``settling_until``。
    失败：成功、失败与用户取消都记，关闭时丢弃的不记；话题已被删除时不记。记不下（磁盘错误）时
        生成控制器只记日志，位置没有推进，这段交互以后会再沉淀一次。

删除（``delete_topic``）
    看什么：话题是否正被占用。
    做什么：取 lease，删除日志与驻留记录，不沉淀；已经接纳的沉淀照常执行，完成时不再记下。
    推进：没有，话题不在了。
    失败：话题正被占用或不存在时报告没有删除（``removed=False``）。

启动（``ShortTermMemoryStore.load``、``restore_topics``、``resettle_retired_topics``）
    看什么：日志。
    做什么：重放日志读回话题池，按最近写入的先后登记驻留，访问时间从读回时开始计算；已经退役、
        沉淀都已完成的日志删除；生成队列启动之后，重新发起退役话题没有完成的最后一次沉淀。
    推进：日志里的四个位置照原样读回，进程内的两个从 0 开始。
    失败：读不回的日志移进 rejected 文件，这个话题不读回；最后一次沉淀没有被接纳时日志保留到
        下次启动。

关闭、崩溃
    看什么：无。
    做什么：关闭时停掉两个扫描与话题维护，交互提交队列排空；生成队列中还没开始写入记忆的沉淀
        丢弃，已经开始写入的写完并记下位置；话题不沉淀、不另外保存（日志随时落盘）。崩溃时
        进程内的一切同样丢失。
    推进：日志停在上一次记下的位置。
    失败：重启后池中的话题由空闲扫描重新沉淀，退役的话题在启动时重新发起最后一次沉淀。

lease
=====

lease 是同一话题的占用权（``TopicWorkingSet.acquire`` 与 ``release``），非阻塞：取不到时以
``TopicBusyError`` 报告，或在扫描中跳过这个话题。它保护先读话题、跨 await 等待、再按读到的内容
写回的操作，不与同一话题的另一个这类操作或一轮交互的追加交错：折叠在等待摘要期间、沉淀在等待
接纳期间一直持有它，写回时读到的前提（折叠的前缀、冻结的交互数、退役时没有漏掉的交互）因此仍然
成立。

取 lease 的是接收交互（追加这一轮时）、折叠、沉淀（整个接纳期间，含随后的退役）与删除。不取
lease 的三项，以及为什么不需要：

- 写进展记录：只往后追加。这一条必须从 ``recorded_until`` 开始，只有写进展记录会推进它，同一
  话题同时至多一项维护，存储在追加时校验这一点；生成期间话题退役或删除时存储找不到它，这一条
  不记。
- 记下本话题生成的记忆：只往一个与交互、折叠和沉淀位置都不相交的集合里加。
- 记下沉淀完成：``settled_until`` 只往前推进。

后两项与沉淀的接纳交错时，存储总是在最新的读取值上套用，谁也不覆盖谁。存储自己的锁
（``ShortTermMemoryStore`` 内部）只在每一次读写时持有，保证“读当前值 → 套用 → 追加 → 替换”不被
另一次读写打断；它不跨 await，与 lease 是两把锁。
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
    """感知业务门面：编排话题的存储、驻留记录与算法，规则见模块说明的触发表。

    ``engine=None`` 表示感知关闭（原 ``NullPerceptionLayer`` 语义）：摄入
    路径无任何存储副作用；维护用例对空存储自然空转。

    话题的建立、沉淀或退役与不沉淀的删除各发一个 ``topic.*`` 事件；事件只是旁路
    观测，发布失败不影响话题的存储与沉淀。
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

    # ========== 接收交互 ==========

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
        """接收一轮交互：把这一轮与首次使用的附件绑定追加到目标话题，返回真实 topic_id。

        看什么：目标是不是新话题；新话题时池是否已满。
        做什么：池满时先让最久没用的话题退役，建立新话题，取 lease 追加这一轮，刷新访问时间；
            不折叠，是否需要折叠由交互提交队列在应用之后经 :meth:`needs_fold` 判断。
        推进：``interaction_count``。
        失败：话题正被占用时抛出 :class:`TopicBusyError`，目标话题已不在池中时抛出
            :class:`ResourceNotFoundError`。追加是最后一个可能失败的步骤：要么失败且这一轮没有
            写入，要么写入并成功；提交队列只允许没有成功的交互重新提交，同一轮因此不会写入两次，
            在追加之后加入可能失败的步骤会破坏这一点。

        目标为 ``NEW_TOPIC`` 时以提交携带的标题与摘要建立话题（缺省时用默认标题），后续路由
        依赖候选话题的标题与摘要。
        """
        if self._engine is None:  # 感知关闭：无任何存储副作用
            return target_topic_id

        logger.debug(
            "PerceptionFamiliar 接收一轮交互: interaction_id=%s target_topic_id=%s",
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

    # ========== 沉淀 ==========

    async def _settle_topic(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        reason: TriggerReason,
        retire: bool,
        raise_on_admission_failure: bool,
    ) -> tuple[bool, MemoryGenerationTask | None]:
        """空闲、池满与手动三个触发共用的沉淀：取 lease → 冻结材料 → 等待接纳 → 记下在途或退役。

        三者只在接纳之后是否退役上分开：只沉淀（``retire=False``）时话题留在池中，不刷新访问
        时间；退役时话题离开话题池。有没有新交互以 ``settled_until`` 与 ``settling_until`` 中较远
        的为准，在途期间同一段不会再沉淀；没有新交互时不生成：只沉淀时什么也不做，退役时直接
        退役。``reason`` 只作为材料的 provenance 标签，不驱动分支。

        接纳之后只记下 ``settling_until``，沉淀到达终态时由 :meth:`record_settled` 推进
        ``settled_until``；协调器没有建任务（材料里没有可生成的轮次）时没有可等的终态，立即推进。
        接纳失败时话题与各位置原样保留，可以重试。

        两把锁：等待接纳的整个期间持有话题的 lease，同一话题的交互、折叠、删除与另一次沉淀都进
        不来；存储的锁只在每一次读写时持有，不跨这次等待。
        返回 (这次沉淀或退役是否完成, 已接纳的生成任务或 None)。
        """
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始沉淀")
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                self._working_set.remove(belong_to, topic_id)  # 清理陈旧驻留条目
                return False, None
            # 冻结材料：字段转换与没有材料的判断收口在数据模型内。
            material = TopicSettleMaterial.from_topic_data(
                topic,
                belong_to=belong_to,
                reason=reason,
                retires_topic=retire,
                settled_until=max(
                    topic.settled_until, self._working_set.settling_until(belong_to, topic_id)
                ),
            )
            if material is None:  # 已沉淀与在途的沉淀覆盖了全部交互，不触发生成
                if not retire:
                    return False, None
                self._retire_topic(belong_to, topic_id, reason)
                self._emit_topic_settled(
                    belong_to, topic_id, reason, generation_submitted=False, retired=True
                )
                return True, None

            # 持有 lease 等待生成队列接纳。
            try:
                generation_task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, material
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
            settling_until = len(material.interactions)
            if generation_task is None:
                self._store.append_settled(belong_to, topic_id, settled_until=settling_until)
            else:
                self._working_set.mark_settling(belong_to, topic_id, settling_until)
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
        """话题退役，离开话题池与驻留；还有没完成的沉淀时写退役记录、保留日志，等最后一次沉淀
        完成再删除。

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
        """手动沉淀并退役指定话题。

        看什么：``settled_until`` 与 ``settling_until`` 中较远的那个之后有没有交互。
        做什么：有新交互先沉淀，然后退役；没有时直接退役（见 :meth:`_settle_topic`）。
        推进：话题退役，不再有池中的位置。
        失败：话题不存在时抛出 :class:`ResourceNotFoundError`；生成队列已满或已停止时抛出
            :class:`TopicSettleAdmissionError`，话题内容保留可重试；目标正被占用是瞬态冲突，
            抛出 :class:`TopicBusyError`。
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
        """工作视图超过折叠阈值且有可折叠的旧前缀时为真；话题不存在时为假。"""
        topic = self._store.get(belong_to, topic_id)
        return topic is not None and bool(self._blocks_to_fold(topic))

    async def fold_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """折叠话题的工作视图（交互提交队列的维护项）。

        看什么：工作视图的 token 是否超过折叠阈值，轮数是否多于保留的最近几轮。
        做什么：取 lease，把工作视图除最近几轮外折叠为折叠摘要，追加一条折叠记录。
        推进：``folded_until``。
        失败：正被占用时抛出 :class:`TopicBusyError` 留待重试；话题已退役或删除时直接结束。
            写回在摘要返回后一次完成，中途取消或失败都不改动话题。
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

        调用方持有该话题的 lease。被折叠的 blocks 仍留在交互记录中，沉淀时随全部记录冻结。
        """
        engine = self._engine
        if engine is None:  # 感知关闭：与接收交互一致，不做任何折叠
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
        """写一条进展记录（交互提交队列的维护项，由停顿扫描排入）。

        看什么：``recorded_until`` 之后有没有交互。
        做什么：不取 lease，以旧进展记录与 ``recorded_until`` 之后的原文生成一段总结，生成完成才
            追加到话题的日志上。
        推进：``recorded_until``，推进到这次生成开始时的 ``interaction_count``。
        失败：生成失败时记下 ``progress_failed_until`` 后照常抛出：同一位置不再重试，否则提供商
            故障时停顿扫描每一轮都会再排入一次。生成期间话题退役或删除时这一条不记；中途取消或
            进程退出什么也不写。
        """
        engine = self._engine
        if engine is None or not engine.writes_progress_record:
            return
        topic = self._store.get(belong_to, topic_id)
        if topic is None:
            return
        start = topic.recorded_until
        covered = topic.interactions[start:]
        if not covered:
            return
        covered_until = topic.interaction_count
        try:
            text = await engine.generate_progress_entry(
                topic.progress_record, covered, first_turn_number=start + 1
            )
        except Exception:
            self._working_set.mark_progress_failed(belong_to, topic_id, covered_until)
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
        """把一次生成执行后本话题新生成的记忆与合并改指记到话题上；不取 lease。

        退役话题的日志删除之前照样记：话题退役时可能还有排在前面的同话题沉淀，随退役执行的
        最后一次沉淀要认得它们写下的记忆。话题已被删除、或日志已经删除时不记。
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
        self, topic_id: str, *, belong_to: WorkspaceIdentity, settled_until: int
    ) -> None:
        """一次沉淀到达终态（生成控制器经本地路由调用）；不取 lease。

        看什么：话题在池中还是已经退役。
        做什么：在日志中记下这次沉淀覆盖到第几轮；话题已经退役、这一次之后再没有未沉淀的交互
            时，这就是最后一次沉淀，随即删除日志。
        推进：``settled_until``（只往前），清除不超过它的 ``settling_until``。
        失败：话题已被删除、或日志已经删除时不记。成功、失败与用户取消都会调用这里，关闭时丢弃
            的沉淀不调用，这由生成控制器决定。
        """
        self._working_set.clear_settling(belong_to, topic_id, settled_until=settled_until)
        retired = self._store.get(belong_to, topic_id) is None
        topic = self._store.append_settled(belong_to, topic_id, settled_until=settled_until)
        if topic is not None and retired and not topic.has_unsettled_interactions:
            self._store.delete(belong_to, topic_id)

    # ========== 退役与删除 ==========

    async def _retire_lru_for_new_topic(
        self, belong_to: WorkspaceIdentity, target_topic_id: str
    ) -> None:
        """池满时为新话题腾位置：让最久没用的话题逐个退役，退役前先沉淀它还没有沉淀的交互。

        看什么：目标是不是新话题；池是否已满；哪个话题最久没用、没有被占用。
        做什么：通常只退役一个；驻留容量在两次运行之间调小时，读回的话题超出容量，这里一次
            退役到池有空位。目标是已有话题时不腾位置，只确认它还在池中。
        推进：被退役的话题不再有池中的位置。
        失败：目标已有话题不在池中时抛出 :class:`ResourceNotFoundError`；候选正被占用时改选其他
            话题，全部正被占用时抛出 :class:`TopicBusyError`；接纳失败的错误原样抛给调用方。
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
        """删除话题（话题池的手动删除）。

        看什么：话题是否正被占用。
        做什么：取 lease，删除日志与驻留记录，不沉淀；已经被接纳的沉淀照常执行，完成时话题已经
            不在，不再记下。
        推进：没有，话题不在了。
        失败：正被占用或话题不存在时报告 ``removed=False``。
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

    # ========== 扫描与启动时读回 ==========

    async def scan_idle_topics_once(self) -> list[str]:
        """空闲扫描（调度任务 ``perception_idle_settle``）；返回被沉淀或退役的话题。

        看什么：访问时间超过 ``idle_timeout_seconds`` 且没有被占用；``settled_until`` 与
            ``settling_until`` 中较远的那个之后有没有交互。
        做什么：有新交互的沉淀，话题留在池中、访问时间不变，没有新交互就不会再被沉淀，同一段
            空闲因此不会反复沉淀；从没接收过交互的空话题（为写入预先建立、这一轮却没有交互）
            留在池中只占位置，直接退役、不生成。
        推进：``settling_until``（见 :meth:`_settle_topic`）。
        失败：接纳失败时只记日志，话题原样保留，下一轮再看；正被占用时这一轮跳过。
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
            except TopicBusyError:  # 列出候选之后才被占用，留给下一轮
                continue
            if completed:
                settled.append(topic_id)
        return settled

    async def scan_paused_topics_once(
        self, schedule_progress_entry: Callable[[WorkspaceIdentity, str], Awaitable[bool]]
    ) -> list[str]:
        """停顿扫描（调度任务 ``perception_progress_record``）；返回排入了写一条进展记录的话题。

        看什么：访问时间超过 ``progress_record_pause_seconds`` 且没有被占用；``recorded_until`` 之后
            有交互；``progress_failed_until`` 小于 ``interaction_count``。没有写进展记录的模型时
            什么都不排。
        做什么：经 ``schedule_progress_entry`` 把一项维护排进交互提交队列，执行见
            :meth:`write_progress_entry`。停顿阈值小于空闲时限，空闲时通常已经写过一条；写下一条
            之后访问时间不变，没有新交互就不会再写。
        推进：这里不推进位置。
        失败：话题已有未结束的维护项、维护已关闭或队列已满时排入函数返回假，下一轮再看。
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
            if topic.recorded_until == topic.interaction_count:
                continue
            if self._working_set.progress_failed_until(belong_to, topic_id) >= (
                topic.interaction_count
            ):
                continue
            if await schedule_progress_entry(belong_to, topic_id):
                scheduled.append(topic_id)
        return scheduled

    def restore_topics(self, topics: Sequence[TopicData]) -> None:
        """启动时为从日志读回的话题登记驻留；在接收交互之前调用，驻留记录此时为空。

        看什么：每个话题最近一次写入的时间（``last_update``）。
        做什么：按 ``last_update`` 从旧到新登记：最近写入的话题最后登记，最久没有写入的是第一个
            退役候选。访问时间不写进日志，这只是上次运行访问顺序的近似。同一 Workspace 的话题
            超过驻留容量（配置在两次运行之间调小）时全部读回，下一次需要新话题时逐个退役，
            未沉淀的交互不丢。
        推进：访问时间从读回时开始计算，``settling_until`` 与 ``progress_failed_until`` 从 0 开始。
        失败：不会失败；读不回的日志在存储读回时已经移进 rejected 文件。
        """
        for topic in sorted(topics, key=lambda topic: topic.last_update):
            self._working_set.touch(topic.workspace_identity, topic.topic_id)

    async def resettle_retired_topics(
        self, retired: Sequence[tuple[TopicData, TopicRetiredRecord]]
    ) -> int:
        """重新发起上次运行中已退役、最后一次沉淀没有完成的话题的沉淀，返回发起的数量。

        看什么：退役话题日志里的 ``settled_until`` 与退役原因。
        做什么：在生成队列启动之后调用。材料由日志组装，与运行时的沉淀一样冻结全部交互，带着
            原来的退役原因；话题仍是退役，最后一次沉淀到达终态时经 :meth:`record_settled` 删除日志。
        推进：生成队列没有为它建任务时直接推进 ``settled_until`` 并删除日志。
        失败：生成队列已满或已停止时日志保留，下次启动时再发起。
        """
        resubmitted = 0
        for topic, retired_record in retired:
            belong_to = topic.workspace_identity
            material = TopicSettleMaterial.from_topic_data(
                topic,
                belong_to=belong_to,
                reason=retired_record.reason,
                retires_topic=True,
                settled_until=topic.settled_until,
            )
            assert material is not None  # 读回时只交来还有未沉淀交互的话题
            try:
                generation_task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, material
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
                    topic.topic_id, belong_to=belong_to, settled_until=len(material.interactions)
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

        ``created_by`` 只用于新建话题的事件：接收交互时建立，或为写入预先确定
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
        """发出 ``topic.settled``：沉淀被生成队列接纳，或话题没有新交互直接退役。

        在接纳时发出，不等沉淀到达终态；它与日志中沉淀到达终态时写下的 ``settled`` 记录不是
        同一件事，``settled_until`` 只由后者推进。``retired`` 表示话题是否随之离开话题池，没有
        新交互的退役不提交生成。
        """
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

        以存储为准：话题不在存储中时不登记驻留，驻留记录里不会多出存储里没有的话题。
        """
        if self._store.get(belong_to, topic_id) is None:
            return
        self._working_set.touch(belong_to, topic_id)


__all__ = ["PerceptionFamiliar"]
