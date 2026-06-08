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
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.engines.generation.models import (
    GenerationContext,
    GenerationRequest,
    MemoryGenerationResult,
)
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.system.runtime.control import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskRegistry,
    MemoryGenerationTaskStatus,
)

logger = logging.getLogger(__name__)


class MemoryGenerationTaskController:
    """
    记忆生成任务控制器。

    这是 Phase 2 任务运行时的集中入口。它不负责感知层路由和生命周期
    gardening，只负责把已经确定要执行的记忆生成请求包装为可观测、可取消、
    可查询的 MemoryGenerationTask。
    """

    def __init__(
        self,
        *,
        storage: Any,
        bus: Optional[Any] = None,
        generation_engine: Optional[Any] = None,
        task_registry: Optional[MemoryGenerationTaskRegistry] = None,
    ) -> None:
        self.storage = storage
        self._bus = bus
        self.generation_engine = generation_engine
        self._task_registry = task_registry or MemoryGenerationTaskRegistry()

    async def run_archive_generation(
        self,
        topic_id: str,
        gen_context: GenerationContext,
    ) -> MemoryGenerationTask:
        """启动被动归档链路的记忆生成任务。"""
        return await self._create_and_run_task(
            topic_id=topic_id,
            label=topic_id,
            source=MemoryGenerationSource.ARCHIVE,
            coro_factory=lambda mt: self._run_archive_task(mt, gen_context),
        )

    async def run_active_generation(
        self,
        tasks: List[PendingAtomMaterializeTask],
        topic_id: str,
        *,
        gen_context: GenerationContext,
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
                    coro_factory=lambda mt, t=task: self._run_active_task(mt, t, gen_context),
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
        这保证上层可以先把 task_id 暴露给前端，再由控制面处理取消和状态观察。
        """
        memory_task = MemoryGenerationTask(
            task_id=str(uuid.uuid4()),
            topic_id=topic_id,
            label=label,
            source=source,
            pending_alias=pending_alias,
        )
        self._task_registry.register(memory_task)

        # 没有生成引擎时仍返回一个已完成 task，保持调用方契约稳定。
        if skip or self.generation_engine is None:
            self._task_registry.close(memory_task.task_id, MemoryGenerationTaskStatus.COMPLETED)
            return memory_task

        bg_task = asyncio.create_task(
            coro_factory(memory_task),
            name=f"memory_task_{memory_task.task_id[:8]}",
        )
        memory_task.attach_task(bg_task)

        def _terminal_status() -> MemoryGenerationTaskStatus:
            if memory_task.cancelled or memory_task.status == MemoryGenerationTaskStatus.CANCELLED:
                return MemoryGenerationTaskStatus.CANCELLED
            if memory_task.status == MemoryGenerationTaskStatus.FAILED:
                return MemoryGenerationTaskStatus.FAILED
            return MemoryGenerationTaskStatus.COMPLETED

        def _mark_task_cancelled() -> None:
            if memory_task.status not in (
                MemoryGenerationTaskStatus.COMPLETED,
                MemoryGenerationTaskStatus.FAILED,
                MemoryGenerationTaskStatus.CANCELLED,
            ):
                memory_task.status = MemoryGenerationTaskStatus.CANCELLED

        def _schedule_terminal_status_publish(status: MemoryGenerationTaskStatus) -> None:
            if memory_task.finished_at is not None or self._bus is None:
                return
            memory_task.status = status
            memory_task.finished_at = datetime.now(timezone.utc)
            asyncio.create_task(
                self._publish_memory_task_status(memory_task),
                name=f"memory_task_status_{memory_task.task_id[:8]}",
            )

        def _done_callback(t: asyncio.Task) -> None:
            # 后台 task 被外部 cancel 时，补齐 registry 与实时状态事件。
            if t.cancelled():
                _mark_task_cancelled()
                _schedule_terminal_status_publish(MemoryGenerationTaskStatus.CANCELLED)
                self._task_registry.close(memory_task.task_id, MemoryGenerationTaskStatus.CANCELLED)
                return

            exc = t.exception()
            if memory_task.cancelled:
                _mark_task_cancelled()
                _schedule_terminal_status_publish(MemoryGenerationTaskStatus.CANCELLED)
                self._task_registry.close(memory_task.task_id, MemoryGenerationTaskStatus.CANCELLED)
            elif exc is not None:
                logger.error(f"memory task {memory_task.task_id} failed: {exc}", exc_info=exc)
                _schedule_terminal_status_publish(MemoryGenerationTaskStatus.FAILED)
                self._task_registry.close(memory_task.task_id, MemoryGenerationTaskStatus.FAILED)
            else:
                self._task_registry.close(memory_task.task_id, _terminal_status())

        bg_task.add_done_callback(_done_callback)
        return memory_task

    async def _run_archive_task(
        self,
        memory_task: MemoryGenerationTask,
        gen_context: GenerationContext,
    ) -> None:
        """执行被动 ARCHIVE 生成链路，并在终态前回填任务结果。"""
        memory_task.status = MemoryGenerationTaskStatus.RUNNING
        if memory_task.cancelled:
            memory_task.status = MemoryGenerationTaskStatus.CANCELLED
            memory_task.finished_at = datetime.now(timezone.utc)
            await self._publish_memory_task_status(memory_task)
            return

        memory_task.started_at = datetime.now(timezone.utc)
        await self._publish_memory_task_status(memory_task)

        try:
            logger.info(f"Memory generation archive task: {len(gen_context.turns)} turns")
            request = GenerationRequest(context=gen_context)
            results = await self._run_generation(request)
            # 被动链路可能产出多条记忆；Phase 2 先回填第一条可用 canonical_alias。
            self._backfill_memory_task_result(memory_task, results)
            memory_task.status = MemoryGenerationTaskStatus.COMPLETED
        except asyncio.CancelledError:
            memory_task.status = MemoryGenerationTaskStatus.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Archive memory generation failed: {e}", exc_info=True)
            memory_task.status = MemoryGenerationTaskStatus.FAILED
            memory_task.error = str(e)
        finally:
            memory_task.finished_at = datetime.now(timezone.utc)
            await self._publish_memory_task_status(memory_task)

    async def _run_active_task(
        self,
        memory_task: MemoryGenerationTask,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
    ) -> None:
        """执行单个 MTP WRITE/UPDATE 主动生成任务。"""
        memory_task.status = MemoryGenerationTaskStatus.RUNNING

        if memory_task.cancelled:
            memory_task.status = MemoryGenerationTaskStatus.CANCELLED
            memory_task.finished_at = datetime.now(timezone.utc)
            await self._publish_pending_atom_cancelled(task.pending_alias)
            await self._publish_memory_task_status(memory_task)
            return

        if memory_task.cancelled:
            memory_task.status = MemoryGenerationTaskStatus.CANCELLED
            memory_task.finished_at = datetime.now(timezone.utc)
            await self._publish_pending_atom_cancelled(task.pending_alias)
            await self._publish_memory_task_status(memory_task)
            return

        memory_task.started_at = datetime.now(timezone.utc)
        await self._publish_memory_task_status(memory_task)

        try:
            if task.source_verb == "WRITE":
                results = await self._run_mode_b(task, gen_context)
            else:
                results = await self._run_mode_c(task, gen_context)
            # 主动链路优先按 pending_alias 匹配结果，避免多 result 时串到其他任务。
            self._backfill_memory_task_result(
                memory_task,
                results,
                pending_alias=task.pending_alias,
            )
            memory_task.status = MemoryGenerationTaskStatus.COMPLETED
        except asyncio.CancelledError:
            memory_task.status = MemoryGenerationTaskStatus.CANCELLED
            await self._publish_pending_atom_cancelled(task.pending_alias)
            raise
        except Exception as e:
            logger.error(
                f"Active memory generation failed: pending_alias={task.pending_alias}, err={e}",
                exc_info=True,
            )
            memory_task.status = MemoryGenerationTaskStatus.FAILED
            memory_task.error = str(e)
            await self._publish_pending_atom_failed(task.pending_alias)
        finally:
            memory_task.finished_at = datetime.now(timezone.utc)
            await self._publish_memory_task_status(memory_task)

    async def _run_mode_b(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
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
        return await self._run_generation(request)

    async def _run_mode_c(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
    ) -> List[MemoryGenerationResult]:
        """Mode C：加载 UPDATE 目标记忆，并转换为 GenerationRequest。"""
        from uuid import UUID as _UUID

        from hivememory.core.models.pending import UpdateFocus

        focus = task.focus
        assert isinstance(focus, UpdateFocus)
        logger.info(f"Mode C UPDATE: alias='{focus.base_alias}'")

        existing_result = self.storage.get_memory(_UUID(focus.base_uuid))
        existing = (
            await existing_result if inspect.isawaitable(existing_result)
            else existing_result
        )
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
        return await self._run_generation(request)

    async def _run_generation(self, request: GenerationRequest) -> List[MemoryGenerationResult]:
        """
        调用生成引擎并发布 PendingAtom settlement 事件。

        返回结构化 MemoryGenerationResult，供任务终态前回填 canonical_alias。
        """
        process_result = self.generation_engine.process(request)
        results = (
            await process_result if inspect.isawaitable(process_result)
            else process_result
        )

        memories = [r.atom for r in results if r.atom is not None]
        logger.info(
            f"Extracted {len(memories)} memories"
            if memories
            else "No memories extracted"
        )

        if self._bus is not None:
            for result in results:
                if result.settlement is not None:
                    # Settlement 是主动 WRITE/UPDATE 与 Alice PendingAtomRuntime 的应答桥。
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

    def get_task(self, task_id: str) -> Optional[MemoryGenerationTask]:
        """按 task_id 查询运行时任务。"""
        return self._task_registry.get(task_id)

    def list_tasks(self) -> List[MemoryGenerationTask]:
        """列出当前 registry 中仍保留的任务。"""
        return self._task_registry.list_all()

    def cancel_task(self, task_id: str) -> bool:
        """请求取消指定记忆生成任务。"""
        return self._task_registry.cancel(task_id)


__all__ = ["MemoryGenerationTaskController"]
