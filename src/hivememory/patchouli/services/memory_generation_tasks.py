"""
记忆生成任务控制器 (Memory Generation Task Controller)

定位：Patchouli 的 Phase 2 运行时任务控制面。

职责：
    - 统一创建、注册和调度 MemoryGenerationTask
    - 承接主动 WRITE/UPDATE 与被动 ARCHIVE 两类记忆生成链路
    - 发布任务状态、PendingAtom 结算/失败/取消事件
    - 在生成结果返回后回填 MemoryGenerationTask.canonical_alias
    - 提供 task 查询与取消 API

该模块从 LibrarianCore 中拆出。LibrarianCore 保留协调入口职责，
本模块专注于“记忆生成任务”的生命周期、可观测性和控制流。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from hivememory.core.models.artifact import ArtifactRef, MemoryProvenance, MemoryVersionSnapshot
from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationRequest,
    MemoryGenerationResult,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskStatus,
    memory_task_to_payload,
)
from hivememory.system.runtime.events import NullRuntimeEventSink, RuntimeEventSink

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({
    MemoryGenerationTaskStatus.COMPLETED,
    MemoryGenerationTaskStatus.CANCELLED,
    MemoryGenerationTaskStatus.FAILED,
})


class MemoryGenerationTaskController:
    """
    记忆生成任务控制器。

    这里的核心约束是：业务协程拥有 MemoryGenerationTask 的生命周期判断权，
    但字段写入只能经过 _start_task / _finish_task 两个入口。asyncio.Task
    本身只负责承载后台执行，不再通过 done_callback 反推 completed/failed/cancelled。

    RuntimeEvent 与本地总线事件都是可观测副作用，必须 best-effort 发布；
    发布失败不能反向改变业务终态，也不能阻止 registry close。
    """

    def __init__(
        self,
        *,
        mid_term: "MidTermMemoryStore",
        bus: Optional[Any] = None,
        generation_engine: Optional[Any] = None,
        task_registry: Optional[MemoryGenerationTaskRegistry] = None,
        runtime_events: RuntimeEventSink | None = None,
        artifact_engine: Optional["ArtifactEngine"] = None,
    ) -> None:
        self._mid_term = mid_term
        self._bus = bus
        self.generation_engine = generation_engine
        self._task_registry = task_registry or MemoryGenerationTaskRegistry()
        self._events = runtime_events or NullRuntimeEventSink()
        self._artifact_engine = artifact_engine

    async def run_archive_generation(
        self,
        topic_id: str,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> MemoryGenerationTask:
        """启动被动归档链路的记忆生成任务。"""
        return await self._create_and_run_task(
            topic_id=topic_id,
            label=topic_id,
            source=MemoryGenerationSource.ARCHIVE,
            coro_factory=lambda mt: self._run_archive_task(mt, gen_context, interaction_ref),
        )

    async def run_active_generation(
        self,
        tasks: List[PendingAtomMaterializeTask],
        topic_id: str,
        *,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationTask]:
        """启动 MTP WRITE/UPDATE 主动链路的记忆生成任务。"""
        memory_tasks = []
        for task in tasks:
            memory_tasks.append(
                await self._create_and_run_task(
                    topic_id=topic_id,
                    label=task.pending_alias,
                    source=MemoryGenerationSource(task.source_verb),
                    pending_alias=task.pending_alias,
                    coro_factory=lambda mt, t=task: self._run_active_task(mt, t, gen_context, interaction_ref),
                )
            )
        return memory_tasks

    async def _create_and_run_task(
        self,
        topic_id: str,
        label: str,
        source: MemoryGenerationSource,
        coro_factory: Callable[[MemoryGenerationTask], Any],
        pending_alias: Optional[str] = None,
        skip: bool = False,
    ) -> MemoryGenerationTask:
        """
        统一任务工厂：创建运行时句柄、注册任务、绑定后台协程。

        返回值必须立即可用，后台生成通过 memory_task._bg_task 继续执行。
        任务协程必须自行调用 _start_task / _finish_task 完成业务生命周期；
        此处只负责调度，不通过 done_callback 发布终态。
        """
        memory_task = MemoryGenerationTask(
            task_id=str(uuid.uuid4()),
            topic_id=topic_id,
            label=label,
            source=source,
            pending_alias=pending_alias,
        )
        self._task_registry.register(memory_task)
        self._emit_memory_task_event(
            RuntimeEventType.MEMORY_TASK_CREATED,
            memory_task,
            message="Memory generation task created.",
        )

        # 没有生成引擎时仍返回一个已完成 task，保持调用方契约稳定。
        if skip or self.generation_engine is None:
            self._task_registry.close(memory_task.task_id, MemoryGenerationTaskStatus.COMPLETED)
            self._emit_memory_task_event(
                RuntimeEventType.MEMORY_TASK_COMPLETED,
                memory_task,
                message="Memory generation task completed without generation engine.",
            )
            return memory_task

        bg_task = asyncio.create_task(
            coro_factory(memory_task),
            name=f"memory_task_{memory_task.task_id[:8]}",
        )
        memory_task.attach_task(bg_task)
        return memory_task

    async def _run_archive_task(
        self,
        memory_task: MemoryGenerationTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> None:
        """
        执行被动 ARCHIVE 生成链路。

        本方法只判断业务结果：成功、取消或失败。RUNNING/终态字段写入、
        RuntimeEvent、本地状态事件和 registry close 都交给 _start_task/_finish_task。
        """
        if memory_task.cancelled:
            await self._finish_task(memory_task, MemoryGenerationTaskStatus.CANCELLED)
            return

        await self._start_task(memory_task)
        try:
            logger.info(f"Memory generation archive task: {len(gen_context.turns)} turns")
            results = await self._run_generation(
                GenerationRequest(context=gen_context),
                source_intent="ARCHIVE",
                interaction_ref=interaction_ref,
            )
            # 被动链路可能产出多条记忆；回填第一条可用 canonical_alias。
            self._backfill_memory_task_result(memory_task, results)
        except asyncio.CancelledError:
            await self._finish_task(memory_task, MemoryGenerationTaskStatus.CANCELLED)
            raise
        except Exception as e:
            logger.error(f"Archive memory generation failed: {e}", exc_info=True)
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.FAILED,
                error=str(e),
            )
        else:
            await self._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)

    async def _run_active_task(
        self,
        memory_task: MemoryGenerationTask,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> None:
        """
        执行单个 MTP WRITE/UPDATE 主动生成任务。

        主动链路额外关联 PendingAtom：COMPLETED 的 settlement 由 _run_generation
        根据结果发布；FAILED/CANCELLED 则由 _finish_task 统一发布 pending atom 事件。
        """
        if memory_task.cancelled:
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.CANCELLED,
                pending_alias=task.pending_alias,
            )
            return

        await self._start_task(memory_task)
        try:
            if task.source_verb == "WRITE":
                results = await self._run_mode_b(task, gen_context, interaction_ref)
            else:
                results = await self._run_mode_c(task, gen_context, interaction_ref)
            # 主动链路优先按 pending_alias 匹配结果，避免多 result 时串到其他任务。
            self._backfill_memory_task_result(
                memory_task,
                results,
                pending_alias=task.pending_alias,
            )
        except asyncio.CancelledError:
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.CANCELLED,
                pending_alias=task.pending_alias,
            )
            raise
        except Exception as e:
            logger.error(
                f"Active memory generation failed: pending_alias={task.pending_alias}, err={e}",
                exc_info=True,
            )
            await self._finish_task(
                memory_task,
                MemoryGenerationTaskStatus.FAILED,
                error=str(e),
                pending_alias=task.pending_alias,
            )
        else:
            await self._finish_task(memory_task, MemoryGenerationTaskStatus.COMPLETED)

    async def _run_mode_b(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """Mode B：将 MTP WRITE 请求转换为 GenerationRequest。"""
        from hivememory.core.models.pending import WriteFocus

        focus = task.focus
        assert isinstance(focus, WriteFocus)
        logger.info(f"Mode B WRITE: content='{focus.content[:50]}...'")
        request = GenerationRequest(
            context=gen_context,
            write_focus=focus,
            identity=task.identity,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )
        return await self._run_generation(request, source_intent="WRITE", interaction_ref=interaction_ref)

    async def _run_mode_c(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """Mode C：加载 UPDATE 目标记忆，并转换为 GenerationRequest。"""
        from uuid import UUID as _UUID

        from hivememory.core.models.pending import UpdateFocus

        focus = task.focus
        assert isinstance(focus, UpdateFocus)
        logger.info(f"Mode C UPDATE: alias='{focus.base_alias}'")

        existing = await self._mid_term.get(_UUID(focus.base_uuid))
        if existing is None:
            logger.error(f"UPDATE target memory not found: {focus.base_uuid}")
            raise RuntimeError(f"UPDATE target memory not found: {focus.base_uuid}")

        request = GenerationRequest(
            context=gen_context,
            update_focus=focus,
            existing_memory=existing,
            identity=task.identity,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )
        return await self._run_generation(request, source_intent="UPDATE", interaction_ref=interaction_ref)

    async def _run_generation(
        self,
        request: GenerationRequest,
        source_intent: str = "WRITE",
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """
        三步流水线：compute → artifacts → persist。

        持久化职责从 GenerationEngine 上移至此处，确保 artifact refs 在
        第一次 upsert 时就已挂载到 MemoryAtom，无需二次写入。
        """
        # Step 1: 纯计算（引擎不再持久化）
        results = await self.generation_engine.process(request)

        memories = [r.atom for r in results if r.atom is not None]
        logger.info(
            f"Extracted {len(memories)} memories"
            if memories
            else "No memories extracted"
        )

        # Step 2: 构建 artifact 并将 refs 挂载到 atom（不持久化）
        await self._build_memory_artifacts(results, request.context, source_intent, interaction_ref)

        # Step 3: 持久化 CREATE/UPDATE 结果（refs 已就位，一次写入）
        for r in results:
            if r.duplicate_decision in (DuplicateDecision.CREATE, DuplicateDecision.UPDATE) and r.atom is not None:
                try:
                    await self._mid_term.upsert(r.atom)
                    logger.info(f"✓ 记忆已存储: '{r.atom.index.title}' (ID: {r.atom.id})")
                except Exception as e:
                    logger.error(f"存储记忆失败: {e}", exc_info=True)
                    raise

        # 发布 settlement 事件（主动链路应答桥）
        if self._bus is not None:
            for result in results:
                if result.settlement is not None:
                    try:
                        await self._bus.publish(
                            PatchouliLocalEvents.PENDING_ATOM_SETTLED,
                            settlement=result.settlement,
                        )
                    except Exception as pub_err:
                        logger.warning(
                            f"Settlement publish failed for "
                            f"{result.settlement.pending_alias}: {pub_err}"
                        )
                        await self._publish_pending_atom_failed(
                            result.settlement.pending_alias
                        )

        return results

    async def _build_memory_artifacts(
        self,
        results: List[MemoryGenerationResult],
        gen_context: GenerationContext,
        source_intent: str,
        interaction_ref: ArtifactRef | None,
    ) -> None:
        """
        Step 2: 构建 artifact 并将 refs/provenance 挂载到 atom。

        不负责持久化——所有 upsert 由调用方 _run_generation Step 3 统一执行。
        """
        if not self._artifact_engine:
            if interaction_ref:
                for r in results:
                    if r.atom is not None and r.duplicate_decision in (DuplicateDecision.CREATE, DuplicateDecision.UPDATE):
                        r.atom.payload.artifacts.refs.append(interaction_ref)
            return

        builder = self._artifact_engine.memory
        src_refs = [interaction_ref] if interaction_ref else []

        for r in results:
            atom = r.atom
            if atom is None:
                continue
            try:
                if r.duplicate_decision == DuplicateDecision.CREATE:
                    bundle = await builder.build_for_create(
                        memory=atom,
                        context=gen_context,
                        source_intent=source_intent,
                        source_artifact_refs=src_refs,
                    )
                    atom.payload.artifacts.refs.extend([
                        bundle.initial_version_ref, bundle.creation_ref,
                    ])
                    if interaction_ref:
                        atom.payload.artifacts.refs.append(interaction_ref)
                    atom.payload.artifacts.provenance.append(MemoryProvenance(
                        action="created",
                        source_intent=source_intent,
                        source_artifacts=src_refs,
                    ))

                elif r.duplicate_decision == DuplicateDecision.UPDATE:
                    version_ref = await builder.build_for_update(
                        memory_after=atom,
                        snapshot_before=r.memory_before_snapshot,
                        update_source="UPDATE",
                        changelog=atom.payload.artifacts.full_history[-1].get("reason") if atom.payload.artifacts.full_history else None,
                        source_artifact_refs=src_refs,
                    )
                    if atom.payload.artifacts.full_history:
                        atom.payload.artifacts.full_history[-1]["artifact_refs"] = [version_ref.model_dump(mode="json")]
                    if interaction_ref:
                        atom.payload.artifacts.refs.append(interaction_ref)
                    atom.payload.artifacts.refs.append(version_ref)
                    atom.payload.artifacts.provenance.append(MemoryProvenance(
                        action="updated",
                        source_intent=source_intent,
                        source_artifacts=src_refs,
                    ))

            except Exception:
                logger.warning(
                    f"Failed to build memory artifacts for {getattr(atom, 'id', '?')}",
                    exc_info=True,
                )

    def _backfill_memory_task_result(
        self,
        memory_task: MemoryGenerationTask,
        results: List[MemoryGenerationResult],
        pending_alias: Optional[str] = None,
    ) -> None:
        """从生成结果中选择 canonical_alias，并写回运行时任务句柄。"""
        canonical_alias = self._select_canonical_alias(
            results,
            pending_alias=pending_alias,
        )
        if canonical_alias:
            memory_task.canonical_alias = canonical_alias

    def _select_canonical_alias(
        self,
        results: List[MemoryGenerationResult],
        pending_alias: Optional[str] = None,
    ) -> Optional[str]:
        """
        选择任务展示用的 canonical_alias。

        优先级：
            1. settlement.canonical_alias
            2. result.canonical_alias
            3. result.atom.get_alias()

        主动链路会优先匹配 pending_alias；被动链路取第一条可用结果。
        """
        candidates = results
        if pending_alias:
            matched = [
                result for result in results
                if result.pending_alias == pending_alias
                or (
                    result.settlement is not None
                    and result.settlement.pending_alias == pending_alias
                )
            ]
            if matched:
                candidates = matched

        for result in candidates:
            if result.settlement is not None and result.settlement.canonical_alias:
                return result.settlement.canonical_alias
            if result.canonical_alias:
                return result.canonical_alias
            if result.atom is not None:
                get_alias = getattr(result.atom, "get_alias", None)
                if callable(get_alias):
                    alias = get_alias()
                    if alias:
                        return alias
        return None

    async def _publish_pending_atom_cancelled(self, pending_alias: str) -> None:
        """发布主动链路 PendingAtom 取消事件。"""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_CANCELLED,
                pending_alias=pending_alias,
            )
        except Exception as pub_err:
            logger.warning(f"CANCELLED event publish error: {pub_err}")

    async def _publish_pending_atom_failed(self, pending_alias: str) -> None:
        """发布主动链路 PendingAtom 失败事件。"""
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias=pending_alias,
            )
        except Exception as pub_err:
            logger.warning(f"FAILED event publish error: {pub_err}")

    async def _publish_memory_task_status(self, memory_task: MemoryGenerationTask) -> None:
        """发布 MemoryGenerationTask 的实时状态快照。"""
        self._emit_memory_task_event(
            self._event_type_for_task_status(memory_task.status),
            memory_task,
        )
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                PatchouliLocalEvents.MEMORY_TASK_ITEM_STATUS,
                task_id=memory_task.task_id,
                pending_alias=memory_task.pending_alias,
                status=memory_task.status.value,
                canonical_alias=memory_task.canonical_alias,
            )
        except Exception as pub_err:
            logger.warning(f"MEMORY_TASK_ITEM_STATUS publish error: {pub_err}")

    async def _start_task(self, memory_task: MemoryGenerationTask) -> None:
        """
        进入 RUNNING 状态并发布运行中快照。

        这是唯一允许写入 RUNNING/started_at 的入口。若任务已处于终态，
        说明取消或失败已经先发生，不能再回退到 running。
        """
        if memory_task.status in _TERMINAL_STATUSES:
            return
        memory_task.status = MemoryGenerationTaskStatus.RUNNING
        if memory_task.started_at is None:
            memory_task.started_at = datetime.now(timezone.utc)
        await self._publish_memory_task_status(memory_task)

    async def _finish_task(
        self,
        memory_task: MemoryGenerationTask,
        status: MemoryGenerationTaskStatus,
        *,
        error: str | None = None,
        pending_alias: str | None = None,
    ) -> None:
        """
        完成 MemoryGenerationTask 的唯一终态入口。

        该方法先落业务事实：status/error/finished_at；再 best-effort 发布
        PendingAtom 和 memory.task 终态事件；最后无论发布是否失败都 close registry。

        第一次终态调用胜出，后续调用 no-op。这样可以防止取消、异常和重复
        cleanup 互相覆盖终态。
        """
        if memory_task._terminal_status_published:
            return
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"Memory task finish status must be terminal: {status}")

        # 先写业务终态，保证后续发布即使失败，查询接口也能看到最终状态。
        memory_task.status = status
        if error is not None:
            memory_task.error = error
        if memory_task.finished_at is None:
            memory_task.finished_at = datetime.now(timezone.utc)

        try:
            # 主动链路的 failed/cancelled PendingAtom 事件跟随任务终态发布。
            # completed 的 settled 事件由 _run_generation 根据生成结果发布。
            if pending_alias is not None:
                if status == MemoryGenerationTaskStatus.CANCELLED:
                    await self._publish_best_effort(
                        self._publish_pending_atom_cancelled(pending_alias),
                        f"pending atom cancel publish failed: {pending_alias}",
                    )
                elif status == MemoryGenerationTaskStatus.FAILED:
                    await self._publish_best_effort(
                        self._publish_pending_atom_failed(pending_alias),
                        f"pending atom failed publish failed: {pending_alias}",
                    )

            await self._publish_best_effort(
                self._publish_memory_task_status(memory_task),
                f"memory task terminal publish failed: {memory_task.task_id}",
            )
        finally:
            # 发布失败或发布过程被取消都不能阻止任务进入 closed/可淘汰状态。
            memory_task._terminal_status_published = True
            self._task_registry.close(memory_task.task_id, status)

    async def _publish_best_effort(self, awaitable, warning: str) -> None:
        """发布可观测副作用。失败只记录日志，不影响业务终态收敛。"""
        try:
            await awaitable
        except asyncio.CancelledError:
            logger.warning(warning, exc_info=True)
        except Exception:
            logger.warning(warning, exc_info=True)

    def get_task(self, task_id: str) -> Optional[MemoryGenerationTask]:
        """按 task_id 查询运行时任务。"""
        return self._task_registry.get(task_id)

    def list_tasks(self) -> List[MemoryGenerationTask]:
        """列出当前 registry 中仍保留的任务。"""
        return self._task_registry.list_all()

    def cancel_task(self, task_id: str) -> bool:
        """请求取消指定记忆生成任务。"""
        memory_task = self._task_registry.get(task_id)
        ok = self._task_registry.cancel(task_id)
        if ok and memory_task is not None:
            self._emit_memory_task_event(
                RuntimeEventType.MEMORY_TASK_CANCEL_REQUESTED,
                memory_task,
                reason="user_requested",
            )
        return ok

    def _emit_memory_task_event(
        self,
        event_type: RuntimeEventType,
        memory_task: MemoryGenerationTask,
        *,
        reason: str | None = None,
        message: str | None = None,
    ) -> None:
        payload = memory_task_to_payload(memory_task, reason=reason)
        self._events.emit(
            RuntimeEvent(
                event_type=event_type,
                task_type="background",
                task_id=memory_task.task_id,
                topic_id=memory_task.topic_id,
                status=memory_task.status.value,
                reason=reason,
                message=message,
                severity="error" if event_type == RuntimeEventType.MEMORY_TASK_FAILED else "info",
                data=payload,
            )
        )

    @staticmethod
    def _event_type_for_task_status(status: MemoryGenerationTaskStatus) -> RuntimeEventType:
        if status == MemoryGenerationTaskStatus.COMPLETED:
            return RuntimeEventType.MEMORY_TASK_COMPLETED
        if status == MemoryGenerationTaskStatus.CANCELLED:
            return RuntimeEventType.MEMORY_TASK_CANCELLED
        if status == MemoryGenerationTaskStatus.FAILED:
            return RuntimeEventType.MEMORY_TASK_FAILED
        return RuntimeEventType.MEMORY_TASK_STATUS


__all__ = ["MemoryGenerationTaskController"]
