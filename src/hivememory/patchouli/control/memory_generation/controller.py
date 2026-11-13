"""Patchouli 记忆生成任务的控制面。"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.components.work_queue import (
    QueuePolicy,
    TaskOutcome,
    UnknownWorkLaneError,
    UnknownWorkPayloadCodecError,
    WorkErrorSnapshot,
    WorkPayloadEncodeError,
    WorkQueueCapacityError,
    WorkQueueError,
    WorkQueueShutdownSummary,
    WorkQueueStoppedError,
    WorkState,
)
from hivememory.engines.generation.models import GenerationRequest
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmission
from hivememory.patchouli.control.memory_generation.events import (
    MemoryTaskEventEmitter,
)
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
    MemoryGenerationTaskStatus,
    memory_task_from_outcome,
    memory_task_from_spec,
)
from hivememory.patchouli.control.memory_generation.queue import (
    MemoryGenerationHandle,
    MemoryGenerationQueue,
    MemoryGenerationResults,
)
from hivememory.patchouli.control.memory_generation.saved_work import (
    RestoredWorkReport,
    SavedWorkFileError,
    SavedWorkReport,
    take_saved_work,
    write_rejected_work,
    write_saved_work,
)
from hivememory.patchouli.control.memory_generation.task_history import (
    take_task_history,
    write_task_history,
)
from hivememory.patchouli.control.pending_atom_settler import PendingAtomSettler
from hivememory.patchouli.runtime.bus import PatchouliBus

logger = logging.getLogger(__name__)

# 关闭时保存到磁盘、下次启动后继续的任务，在本进程内以这个原因取消。
SHUTDOWN_PERSISTED_REASON = "shutdown_persisted"


class _MemoryGenerationIntentConflictError(ValueError):
    """同一 Active intent 被重复用于不同任务规范。"""


@dataclass
class _MemoryTaskEntry:
    """控制器持有的领域投影元数据。

    ``WorkRecord`` 仍是唯一执行状态源；此处只关联不可变工作定义、类型化句柄、
    对外终态快照以及发布侧的同步原语。
    """

    task_id: str
    spec: MemoryGenerationTaskSpec
    created: MemoryGenerationTask
    handle: MemoryGenerationHandle | None = None
    finalizer: asyncio.Task[MemoryGenerationTask] | None = None
    final_snapshot: MemoryGenerationTask | None = None
    running_published: bool = False
    finalize_lock: asyncio.Lock | None = None

    def lock(self) -> asyncio.Lock:
        """按需创建单任务锁，串行化取消与终态 finalize。"""

        if self.finalize_lock is None:
            self.finalize_lock = asyncio.Lock()
        return self.finalize_lock


class MemoryGenerationTaskController:
    """提交结构化工作，并对外提供只读领域快照。

    通用队列负责接纳、执行与终态；控制器负责把 ``WorkRecord`` 投影为
    ``MemoryGenerationTask``，并在队列终态确定后独立完成 settlement 与领域
    状态事件发布。
    """

    def __init__(
        self,
        *,
        bus: PatchouliBus,
        runtime_events: RuntimeEventPublisher | None = None,
        memory_queue: MemoryGenerationQueue | None = None,
        queue_policy: QueuePolicy | None = None,
        pending_atom_settler: PendingAtomSettler | None = None,
    ) -> None:
        self._bus = bus
        self._pending_atom_settler = pending_atom_settler or PendingAtomSettler(bus)
        events = runtime_events or RuntimeEventPublisher(NullRuntimeEventSink())
        self._task_events = MemoryTaskEventEmitter(events)
        self._queue = memory_queue or MemoryGenerationQueue(
            self._execute_generation,
            runtime_events=events,
            policy=queue_policy,
        )

        self._entries: OrderedDict[str, _MemoryTaskEntry] = OrderedDict()
        # 上次运行结束的任务快照，按结束时间从早到晚：只供任务列表与单个任务的读取，
        # 不能取消，也不参与接纳的幂等判重；同一任务 ID 以本进程的任务为准。
        self._history: OrderedDict[str, MemoryGenerationTask] = OrderedDict()
        self._submission_lock = asyncio.Lock()
        # 话题 ID → 该话题的主动写入（WRITE/UPDATE，以及它们被合并后的保留对象）
        # 已结算到的正式记忆 ID。只在进程内：话题结算时取出并清除；进程崩溃时随进程
        # 丢失，生成退回不区分来源的查重。话题未经结算即被删除，或结算开始后才完成的
        # 同话题写入，留下的记录保留到进程结束。
        self._topic_write_memory_ids: dict[str, set[str]] = {}

    @property
    def queue(self) -> MemoryGenerationQueue:
        """访问记忆生成业务队列。"""

        return self._queue

    async def start(self) -> None:
        """启动记忆生成 lane。"""

        await self._queue.start()

    async def stop(self) -> WorkQueueShutdownSummary:
        """停止队列，并给已确定的领域终态一个短暂收敛窗口。"""

        summary = await self._queue.stop()
        # queue 先完成 running work 的 drain/cancel，再等待领域 finalize 发布终态，
        # 避免随后卸载 local routes 时丢失取消或失败快照。
        await self.wait_all(timeout=0.5)
        return summary

    async def submit_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """提交一个不可变工作定义，并返回其接纳快照。"""

        # 串行化 admission，确保并发重提交不会在首个请求尚未确定入队结果时
        # 误读一个只有 created、尚无 handle 的中间 entry。
        async with self._submission_lock:
            return await self._submit_generation_locked(spec)

    async def _submit_generation_locked(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationTask:
        """在 admission 锁内接纳或复用一项生成任务。"""

        # Active WRITE/UPDATE 的 intent_id 来自 PendingAtom，跨重复 dispatch 保持稳定。
        # 其他来源没有可复用的业务意图，仍使用一次性 task_id。
        task_id = self._task_id_for_spec(spec)
        existing = self._entries.get(task_id)
        if existing is not None:
            if existing.spec != spec:
                raise _MemoryGenerationIntentConflictError(
                    f"memory generation intent already exists with different spec: {task_id}"
                )
            # 幂等重提交只返回原任务的当前投影，不再次发布 created、入队或启动执行。
            return await self._snapshot_entry(existing)

        created = memory_task_from_spec(
            task_id,
            spec,
            created_at=datetime.now(UTC),
        )

        if not self._queue.started:
            raise WorkQueueStoppedError(
                "memory generation queue must be started before submitting work"
            )

        # 只有队列确认接纳后才建立 entry 和发布 created，避免把 admission
        # rejection 混入任务状态机。
        handle = await self._queue.submit(task_id, spec)

        entry = _MemoryTaskEntry(
            task_id=task_id,
            spec=spec,
            created=created,
            handle=handle,
        )
        self._entries[task_id] = entry
        self._task_events.created(created)

        entry.finalizer = asyncio.create_task(
            self._finalize(entry),
            name=f"memory_task_finalize_{task_id[:8]}",
        )
        # 返回接纳时刻的值对象；后续状态变化通过查询、等待和事件获得，不原地
        # 修改这个快照。
        return created

    @staticmethod
    def _task_id_for_spec(spec: MemoryGenerationTaskSpec) -> str:
        """为任务生成稳定身份；Active intent 使用业务 intent_id。"""

        if (
            spec.source
            in {
                MemoryGenerationSource.WRITE,
                MemoryGenerationSource.UPDATE,
            }
            and spec.intent_id
        ):
            return f"active:{spec.intent_id}"
        return str(uuid.uuid4())

    async def admit_generation(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> MemoryGenerationAdmission:
        """接纳一项生成任务并分类结果，供需要区分拒绝与结果未知的提交方使用。

        确定性拒绝由 Controller 以 FAILED 结算关联的 ``PendingAtom``，不伪造 FAILED
        任务；无法判断是否接纳的异常与 intent 载荷冲突都是结果未知，不发布结算事件：
        冲突时既有任务仍拥有该 intent 的结算权。
        """

        async with self._submission_lock:
            try:
                task = await self._submit_generation_locked(spec)
            except asyncio.CancelledError:
                raise
            except _MemoryGenerationIntentConflictError as exc:
                # 已有任务仍拥有该 intent 的最终结算权；冲突重放不能抢先把
                # 同一个 PendingAtom 标记为失败。
                logger.warning(
                    "Memory generation intent payload conflict: "
                    "pending_alias=%s, intent_id=%s, err=%s",
                    spec.pending_alias,
                    spec.intent_id,
                    exc,
                )
                return MemoryGenerationAdmission.unknown()
            except (
                WorkPayloadEncodeError,
                WorkQueueCapacityError,
                WorkQueueStoppedError,
                UnknownWorkLaneError,
                UnknownWorkPayloadCodecError,
            ) as exc:
                logger.warning(
                    "Memory generation admission rejected: "
                    "pending_alias=%s, intent_id=%s, err=%s",
                    spec.pending_alias,
                    spec.intent_id,
                    exc,
                )
            except Exception:
                logger.warning(
                    "Memory generation admission outcome unknown: "
                    "pending_alias=%s, intent_id=%s",
                    spec.pending_alias,
                    spec.intent_id,
                    exc_info=True,
                )
                return MemoryGenerationAdmission.unknown()
            else:
                return MemoryGenerationAdmission.accepted(task)

        # 功能事件发布不占用 admission 锁，避免慢订阅者阻塞其他提交方。
        if spec.pending_alias is not None and spec.intent_id is not None:
            await self._pending_atom_settler.failed(spec.pending_alias, spec.intent_id)
        return MemoryGenerationAdmission.rejected()

    async def _execute_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        on_commit_started: Callable[[], None],
    ) -> list[MemoryGenerationResult]:
        """由队列 handler 调用现有的记忆生成数据面。

        WRITE 与 SETTLE 执行时带上本话题主动写入已结算到的记忆，数据面返回后立即记下
        本次结算到的记忆。在执行时而不是接纳时读取：同一话题先后派发的写入，前一条
        完成后，后一条执行时才能看到它。生成 lane 不按话题排序：结算开始时仍在执行的
        同话题写入不在记录中，它与被动草稿之间照旧按查重处理。话题结算之后不再接收
        写入，SETTLE 成功后清除这个话题的记录；SETTLE 在关闭时被中止的，记录留给
        保存文件中的这项结算。
        """

        if spec.source in {MemoryGenerationSource.SETTLE, MemoryGenerationSource.WRITE}:
            written = set(self._topic_write_memory_ids.get(spec.topic_id, set()))
        else:
            written = set()
        if written:
            spec = _with_topic_writes(spec, written)
        results: list[MemoryGenerationResult] = await self._bus.request(
            PatchouliLocalRoutes.GENERATION_EXECUTE_SPEC,
            spec,
            on_commit_started=on_commit_started,
        )
        if spec.source == MemoryGenerationSource.SETTLE:
            self._topic_write_memory_ids.pop(spec.topic_id, None)
        self._record_topic_writes(spec.topic_id, results)
        return results

    async def _finalize(self, entry: _MemoryTaskEntry) -> MemoryGenerationTask:
        """观察执行开始与队列终态，并完成一次领域终态结算。

        started 信号只用于及时发布 ``RUNNING``，不参与状态判定；最终状态始终
        取自 handle 返回的 ``WorkRecord``。该协程独立于发起请求的客户端生命周期。
        """

        handle = entry.handle
        if handle is None:
            raise RuntimeError("accepted memory work is missing its task handle")

        outcome_waiter = asyncio.create_task(
            handle.wait(),
            name=f"memory_task_outcome_{entry.task_id[:8]}",
        )
        started_waiter = asyncio.create_task(
            handle.wait_started(),
            name=f"memory_task_started_{entry.task_id[:8]}",
        )
        try:
            # 同时等待开始信号和终态，既避免轮询，也覆盖工作在观察前已经快速
            # 完成的情况；无论哪一个先到，最终都必须等待 outcome。
            done, _ = await asyncio.wait(
                {outcome_waiter, started_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if started_waiter in done:
                current = await handle.snapshot()
                if current is not None:
                    await self._publish_running_if_needed(entry, current)
            outcome = await outcome_waiter
        finally:
            for waiter in (outcome_waiter, started_waiter):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(
                outcome_waiter,
                started_waiter,
                return_exceptions=True,
            )

        async with entry.lock():
            # 与 cancel_task 共用单任务锁，确保取消请求和领域终态事件不会交错
            # 覆盖彼此的最终快照。
            snapshot = await self._settle_outcome(entry, outcome)
            entry.final_snapshot = snapshot
            self._retain_terminal_entries()
            return snapshot

    async def _settle_outcome(
        self,
        entry: _MemoryTaskEntry,
        outcome: TaskOutcome[MemoryGenerationResults] | None,
    ) -> MemoryGenerationTask:
        """把队列终态投影为领域终态，并执行终态后的独立副作用。"""

        if outcome is None:
            snapshot = entry.created.as_failed(
                "memory generation work record missing",
                finished_at=datetime.now(UTC),
            )
            await self._publish_terminal(entry, snapshot)
            return snapshot

        await self._publish_running_if_needed(entry, outcome)

        record = outcome.record
        snapshot = memory_task_from_outcome(
            entry.created,
            outcome,
            expose_terminal=True,
        )

        if record.state == WorkState.SUCCEEDED:
            results = list(outcome.result or ())
            # 队列成功已经锁定 chat/生成执行终态；settlement 和领域事件属于后续
            # finalize，失败只影响可观测副作用，不回写通用 work 状态。
            await self._publish_settlements(results)
            await self._submit_nominated_merges(entry.spec, results)
        # 失败已由生成队列的分类器按原因码记录日志；这里只把原因码带进领域终态事件。
        failure = (
            record.last_error if snapshot.status == MemoryGenerationTaskStatus.FAILED else None
        )
        await self._publish_terminal(entry, snapshot, failure=failure)
        return snapshot

    async def _submit_nominated_merges(
        self, spec: MemoryGenerationTaskSpec, results: list[MemoryGenerationResult]
    ) -> None:
        """首次持久化与结算成功后提交后续合并；队列拒绝只留下两条正式记忆。"""
        for result in results:
            if result.merge_nomination is None:
                continue
            merge_spec = MemoryGenerationTaskSpec(
                belong_to=spec.belong_to,
                from_actor=spec.from_actor,
                topic_id=spec.topic_id,
                label=f"Merge: {spec.label}",
                source=MemoryGenerationSource.MERGE,
                request=GenerationRequest(merge_focus=result.merge_nomination),
            )
            try:
                await self.submit_generation(merge_spec)
            except (WorkQueueStoppedError, WorkQueueCapacityError):
                logger.warning(
                    "Memory merge submission rejected: topic_id=%s, retained=%s, absorbed=%s",
                    spec.topic_id,
                    result.merge_nomination.retained_memory_id,
                    result.merge_nomination.absorbed_memory_id,
                    exc_info=True,
                )

    async def get_task(self, task_id: str) -> MemoryGenerationTask | None:
        """按任务标识查询当前只读领域快照，本进程没有时查读回的历史。"""

        entry = self._entries.get(task_id)
        if entry is None:
            return self._history.get(task_id)
        return await self._snapshot_entry(entry)

    async def list_tasks(self) -> list[MemoryGenerationTask]:
        """列出读回的历史与本进程仍保留的任务快照，同一任务 ID 以本进程的为准。"""

        snapshots = dict(self._history)
        for task_id, entry in self._entries.items():
            snapshots[task_id] = await self._snapshot_entry(entry)
        return list(snapshots.values())

    async def _snapshot_entry(self, entry: _MemoryTaskEntry) -> MemoryGenerationTask:
        """从终态缓存或最新 ``WorkRecord`` 构造领域快照。"""

        if entry.final_snapshot is not None:
            return entry.final_snapshot
        if entry.handle is None:
            return entry.created
        outcome = await entry.handle.snapshot()
        if outcome is None:
            return entry.created.as_failed(
                "memory generation work record missing",
            )
        return memory_task_from_outcome(
            entry.created,
            outcome,
            expose_terminal=False,
        )

    async def wait_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> MemoryGenerationTask | None:
        """等待指定任务并返回最新快照；不存在时返回 ``None``。

        超时只停止等待，不取消后台任务；此时返回值仍是该任务当前的非终态
        ``MemoryGenerationTask``，调用方无需再解包一层等待结果。
        """

        entry = self._entries.get(task_id)
        if entry is None:
            return None
        if entry.final_snapshot is not None:
            return entry.final_snapshot
        if entry.finalizer is None:
            return await self._snapshot_entry(entry)
        try:
            if timeout is None:
                snapshot = await asyncio.shield(entry.finalizer)
            else:
                snapshot = await asyncio.wait_for(
                    asyncio.shield(entry.finalizer),
                    timeout=max(0.0, timeout),
                )
        except TimeoutError:
            return await self._snapshot_entry(entry)
        return snapshot

    async def wait_many(
        self,
        task_ids: list[str],
        timeout: float | None = None,
    ) -> list[MemoryGenerationTask | None]:
        """在共享超时窗口内按请求顺序返回任务最新快照。"""

        if not task_ids:
            return []

        waiters = [
            asyncio.create_task(self.wait_task(task_id), name=f"memory_task_wait_{task_id[:8]}")
            for task_id in task_ids
        ]
        done, pending = await asyncio.wait(waiters, timeout=timeout)

        results: list[MemoryGenerationTask | None] = []
        for task_id, waiter in zip(task_ids, waiters):
            if waiter in done:
                results.append(waiter.result())
                continue
            waiter.cancel()
            entry = self._entries.get(task_id)
            if entry is None:
                results.append(None)
            else:
                results.append(await self._snapshot_entry(entry))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return results

    async def wait_all(
        self,
        timeout: float | None = None,
    ) -> list[MemoryGenerationTask]:
        """等待调用时尚未终结的任务。"""

        task_ids = [
            task_id
            for task_id, entry in self._entries.items()
            if entry.final_snapshot is None and entry.finalizer is not None
        ]
        results = await self.wait_many(task_ids, timeout=timeout)
        # task_ids 来自当前 entry 集合，因此此处不会产生缺失项；过滤只用于
        # 保持返回类型精确，并防御未来 retention 策略在等待期间发生变化。
        return [task for task in results if task is not None]

    async def cancel_task(
        self,
        task_id: str,
        *,
        reason: str = "user_requested",
    ) -> bool:
        """请求取消指定任务，并发布被接纳的取消请求快照。"""

        entry = self._entries.get(task_id)
        if entry is None or entry.handle is None or entry.final_snapshot is not None:
            return False
        async with entry.lock():
            if entry.final_snapshot is not None:
                return False
            accepted = await entry.handle.cancel(reason=reason)
            if accepted:
                current_outcome = await entry.handle.snapshot()
                if current_outcome is not None:
                    await self._publish_running_if_needed(entry, current_outcome)
                snapshot = (await self._snapshot_entry(entry)).with_cancel_request(reason)
                self._task_events.cancel_requested(
                    snapshot,
                    reason=reason,
                )
            return accepted

    async def cancel_many(
        self,
        task_ids: list[str],
        *,
        reason: str = "user_requested",
    ) -> int:
        """逐项请求取消并返回运行时接纳的数量。"""

        return sum([await self.cancel_task(task_id, reason=reason) for task_id in task_ids])

    # ========== 关闭时保存、启动时读回 ==========

    async def stop_dispatch(self) -> None:
        """停止取新工作；已在执行的任务照常继续，入队照常接受。"""

        await self._queue.stop_dispatch()

    async def save_unfinished(self, path: Path) -> SavedWorkReport:
        """把还没执行的工作写进保存文件，供下次启动读回；在 ``stop_dispatch`` 之后调用。

        在跑的任务中，还没开始写入记忆的立即取消，按未执行保存，下次从头执行；已经开始
        写入的等它写完，结算与提名的合并照常完成，提名的合并入队后一并保存。保存的任务
        随后以取消结束，原因为 ``shutdown_persisted``。
        """
        running: list[_MemoryTaskEntry] = []
        for entry in list(self._entries.values()):
            if await self._work_state(entry) == WorkState.RUNNING:
                running.append(entry)
        for entry in running:
            if not self._handle(entry).commit_started:
                await self.cancel_task(entry.task_id, reason=SHUTDOWN_PERSISTED_REASON)
        # 等到结算与合并提名都完成：已在写入的任务提名的合并在这期间入队。
        await self.wait_many([entry.task_id for entry in running])

        # 取消与开始写入之间没有原子性：取消送达前已开始写入的任务会写完再结束，
        # 因此以是否开始写入、而不是是否被取消，判断要不要重跑。
        interrupted = {entry.task_id for entry in running if not self._handle(entry).commit_started}
        to_save: list[_MemoryTaskEntry] = []
        for entry in list(self._entries.values()):
            if entry.task_id in interrupted or await self._work_state(entry) in {
                WorkState.QUEUED,
                WorkState.RETRY_WAIT,
            }:
                to_save.append(entry)

        file_written = False
        if to_save:
            items = [
                self._queue.encode_for_restart(entry.task_id, self._spec_to_save(entry.spec))
                for entry in to_save
            ]
            try:
                write_saved_work(path, items)
                file_written = True
            except OSError:
                logger.error(
                    "保存未执行的记忆生成工作失败: reason=saved_work_write_failed, "
                    "path=%s, count=%d",
                    path,
                    len(items),
                    exc_info=True,
                )
            queued_ids = [entry.task_id for entry in to_save if entry.task_id not in interrupted]
            await self.cancel_many(queued_ids, reason=SHUTDOWN_PERSISTED_REASON)
            await self.wait_many(queued_ids)

        return SavedWorkReport(
            saved_by_source=dict(Counter(entry.spec.source.value for entry in to_save)),
            waited_for_commit=len(running) - len(interrupted),
            file_written=file_written,
        )

    async def restore_saved(self, path: Path) -> RestoredWorkReport:
        """读回保存文件并重新接纳其中的工作；在队列启动之后调用，不等待执行。

        读不回或接纳被拒绝的项写进保存文件旁边的 rejected 文件，其余照常接纳。
        """
        try:
            items = take_saved_work(path)
        except SavedWorkFileError as error:
            logger.error(
                "读回保存的记忆生成工作失败: reason=saved_work_unreadable, rejected_file=%s",
                error.rejected_path,
                exc_info=True,
            )
            return RestoredWorkReport(rejected_file=str(error.rejected_path))
        if items is None:
            return RestoredWorkReport()

        rejected: list[object] = []
        for item in items:
            try:
                await self.submit_generation(self._queue.decode_for_restart(item))
            except (TypeError, ValueError, WorkQueueError):
                logger.warning(
                    "保存的记忆生成工作无法重新接纳: reason=saved_work_item_rejected",
                    exc_info=True,
                )
                rejected.append(item)
        rejected_file = None
        if rejected:
            rejected_file = str(write_rejected_work(path, rejected))
            logger.error(
                "部分保存的记忆生成工作没有重新接纳: reason=saved_work_items_rejected, "
                "rejected=%d, rejected_file=%s",
                len(rejected),
                rejected_file,
            )
        return RestoredWorkReport(
            read=len(items),
            resubmitted=len(items) - len(rejected),
            rejected=len(rejected),
            rejected_file=rejected_file,
        )

    def restore_task_history(self, path: Path, *, retention: timedelta) -> None:
        """读回上次正常关闭时写盘的已结束任务；在读回保存的生成工作之前调用。

        只保留结束于 ``retention`` 之内的最近若干条（与已结束任务的保留条数相同）。
        """
        cutoff = datetime.now(UTC) - retention
        tasks = sorted(
            (task for task in take_task_history(path) if _finished_at(task) > cutoff),
            key=_finished_at,
        )
        for task in tasks[-self._queue.terminal_retention :]:
            self._history[task.task_id] = task

    def save_task_history(self, path: Path, *, retention: timedelta) -> None:
        """把已结束的任务写盘，供下次启动读回；在生成队列停止之后调用。

        关闭时保存的工作（取消原因 ``shutdown_persisted``）不算已结束：它在下次启动后
        继续执行，写入与更新以同一任务 ID 重新出现在列表中。只写结束于 ``retention``
        之内的最近若干条；没有可写的任务时不写文件，写盘失败只记日志。
        """
        cutoff = datetime.now(UTC) - retention
        finished = dict(self._history)
        for task_id, entry in self._entries.items():
            snapshot = entry.final_snapshot
            if snapshot is not None and snapshot.cancel_reason != SHUTDOWN_PERSISTED_REASON:
                finished[task_id] = snapshot
        tasks = sorted(
            (task for task in finished.values() if _finished_at(task) > cutoff),
            key=_finished_at,
        )[-self._queue.terminal_retention :]
        if not tasks:
            return
        try:
            write_task_history(path, tasks)
        except OSError:
            logger.error(
                "记忆生成任务历史写盘失败: reason=task_history_write_failed, path=%s, count=%d",
                path,
                len(tasks),
                exc_info=True,
            )

    def _spec_to_save(self, spec: MemoryGenerationTaskSpec) -> MemoryGenerationTaskSpec:
        """结算任务带上本话题主动写入的记录：它只在内存中，执行时据此不改写这些记忆。"""

        written = self._topic_write_memory_ids.get(spec.topic_id)
        if spec.source != MemoryGenerationSource.SETTLE or not written:
            return spec
        return _with_topic_writes(spec, written)

    @staticmethod
    def _handle(entry: _MemoryTaskEntry) -> MemoryGenerationHandle:
        if entry.handle is None:
            raise RuntimeError("accepted memory work is missing its task handle")
        return entry.handle

    async def _work_state(self, entry: _MemoryTaskEntry) -> WorkState | None:
        """任务在队列中的当前状态；领域终态已经确定的任务返回 ``None``。"""

        if entry.final_snapshot is not None:
            return None
        outcome = await self._handle(entry).snapshot()
        return outcome.record.state if outcome is not None else None

    def _retain_terminal_entries(self) -> None:
        """按队列保留策略淘汰最早的领域终态及进程内结果，读回的历史计入同一上限、先淘汰。"""

        terminal_ids = [
            task_id for task_id, entry in self._entries.items() if entry.final_snapshot is not None
        ]
        excess = len(terminal_ids) + len(self._history) - self._queue.terminal_retention
        while excess > 0 and self._history:
            self._history.popitem(last=False)
            excess -= 1
        for task_id in terminal_ids[: max(0, excess)]:
            entry = self._entries.pop(task_id)
            if entry.handle is not None:
                self._queue.release(entry.handle)

    async def _publish_running_if_needed(
        self,
        entry: _MemoryTaskEntry,
        outcome: TaskOutcome[MemoryGenerationResults],
    ) -> None:
        """在首次执行已经开始后至多发布一次运行中快照。"""

        if entry.running_published or outcome.record.started_at is None:
            return

        entry.running_published = True
        running = memory_task_from_outcome(
            entry.created,
            outcome,
            expose_terminal=False,
        )
        self._task_events.running(running)

    async def _publish_terminal(
        self,
        entry: _MemoryTaskEntry,
        snapshot: MemoryGenerationTask,
        *,
        failure: WorkErrorSnapshot | None = None,
    ) -> None:
        """以 best-effort 方式发布领域终态及关联 PendingAtom 事件。

        ``failure`` 是失败工作记录的错误快照：失败事件的 reason 取它的原因码，
        载荷带异常类名；其余终态的 reason 是取消原因。
        """

        pending_alias = entry.spec.pending_alias
        intent_id = entry.spec.intent_id
        if pending_alias is not None and intent_id is not None:
            if snapshot.status == MemoryGenerationTaskStatus.CANCELLED:
                # 关闭时保存的写入在下次启动后还会结算，不是被取消的写入。
                if snapshot.cancel_reason != SHUTDOWN_PERSISTED_REASON:
                    await self._pending_atom_settler.cancelled(pending_alias, intent_id)
            elif snapshot.status == MemoryGenerationTaskStatus.FAILED:
                await self._pending_atom_settler.failed(
                    pending_alias,
                    intent_id,
                    reason=failure.message if failure is not None else None,
                )

        self._task_events.terminal(
            snapshot,
            reason=failure.message if failure is not None else snapshot.cancel_reason,
            error_class=failure.error_class if failure is not None else None,
        )

    def _record_topic_writes(self, topic_id: str, results: list[MemoryGenerationResult]) -> None:
        """记下主动写入结算到的正式记忆，供同一话题之后的 WRITE 与 SETTLE 查重使用。

        带 settlement 的结果只来自主动写入链路：WRITE、UPDATE，以及吸收了写入新建
        记忆的合并（结算改指保留的那条）。
        """
        for result in results:
            if result.settlement is None or result.settlement.canonical_uuid is None:
                continue
            self._topic_write_memory_ids.setdefault(topic_id, set()).add(
                result.settlement.canonical_uuid
            )

    async def _publish_settlements(
        self,
        results: list[MemoryGenerationResult],
    ) -> None:
        """发布主动链路的 ``PendingAtom`` settlement 事件。"""

        for result in results:
            if result.settlement is None:
                continue
            if result.absorbed_memory_uuid is not None:
                await self._pending_atom_settler.merged(
                    result.settlement, absorbed_memory_uuid=result.absorbed_memory_uuid
                )
            else:
                await self._pending_atom_settler.settled(result.settlement)


def _finished_at(task: MemoryGenerationTask) -> datetime:
    """已结束任务的结束时间；领域终态总带结束时间，缺少时以创建时间排序。"""
    return task.finished_at or task.created_at


def _with_topic_writes(
    spec: MemoryGenerationTaskSpec, written: set[str]
) -> MemoryGenerationTaskSpec:
    """把本话题主动写入结算到的记忆并入任务规范；读回的结算任务已带有关闭时的记录。"""

    ids = spec.request.topic_write_memory_ids | frozenset(written)
    request = spec.request.model_copy(update={"topic_write_memory_ids": ids})
    return dataclasses.replace(spec, request=request)


__all__ = ["SHUTDOWN_PERSISTED_REASON", "MemoryGenerationTaskController"]
