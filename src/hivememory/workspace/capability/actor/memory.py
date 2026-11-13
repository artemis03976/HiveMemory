"""actor API：逐次检查操作权限，读取视图交付与写入意图登记属于 workspace。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import UUID

from hivememory.components.cancellation import join_shielded
from hivememory.core.access import WorkspaceOperation
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    MemoryAtom,
    MemoryType,
    PendingAtom,
    ReferenceResolution,
    UpdateFocus,
    WriteFocus,
)
from hivememory.core.protocol.models import RetrievalRequest
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntent
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmissionStatus
from hivememory.workspace.contracts.errors import (
    IntentDispatchRejectedError,
    IntentDispatchUnknownError,
    PendingUpdateNotAllowedError,
)
from hivememory.workspace.contracts.operations import (
    ResolveReferencesRequest,
    ResolveRunTargetRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.thread_ports import ExecutionThreadTable

if TYPE_CHECKING:
    from hivememory.components.bus.global_bus import GlobalSystemBus
    from hivememory.core.access import WorkingActor
    from hivememory.core.models import IdentityScope
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.contracts.threads import ThreadHandle
    from hivememory.workspace.resolution.memory import MemoryReader

logger = logging.getLogger(__name__)

#: 检索命中的来源标签，只用于生命周期事件的观测。
_SEARCH_HIT_SOURCE = "workspace.search"

# 请求级提交的派发话题回调：没有线程的入口在授权之后才为这一轮确定话题。
DispatchTopic = Callable[[], Awaitable[str]]

#: 派发结局不明时，以同一意图重试提交之前的等待秒数；重试次数即元素个数。
_DISPATCH_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)


class ActorMemoryService:
    """actor 资源入口：每个方法在入口以工作 actor 授权，转为本次操作的 scope。

    工作 actor 由操作入口交来：线程提交取自线程表的解析结果，请求级提交取自
    入口的认证结果。目标取工作 actor 的驻留 Workspace。
    """

    def __init__(
        self,
        global_bus: GlobalSystemBus,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
        memory_reader: MemoryReader,
        threads: ExecutionThreadTable,
    ) -> None:
        self._global_bus = global_bus
        self._authorizer = operation_authorizer
        self._reader = memory_reader
        self._threads = threads
        # 提交与 L0 回读必须是同一份登记：只经读取视图取得，不另行注入。
        self._intents = memory_reader.intents

    async def submit_write_intent(
        self,
        request: SubmitWriteIntentRequest,
        *,
        actor: WorkingActor,
        thread: ThreadHandle | None = None,
        dispatch_topic: DispatchTopic | None = None,
    ) -> PendingAtom:
        """提交 WRITE（``memory.write``），生成队列接纳后才返回 ACK。

        ``thread`` 与 ``dispatch_topic`` 恰有其一，见 :meth:`_dispatch_topic`。
        """
        scope = self._authorize(actor, WorkspaceOperation.MEMORY_WRITE)
        topic_id = await self._dispatch_topic(thread, dispatch_topic)
        process_id, thread_id = self._resolve_thread(thread)
        pending = self._intents.register_write(
            request.focus,
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
            thread_id=thread_id,
        )
        return await self._dispatch(pending, topic_id=topic_id, scope=scope)

    async def submit_update_intent(
        self,
        request: SubmitUpdateIntentRequest,
        *,
        actor: WorkingActor,
        thread: ThreadHandle | None = None,
        dispatch_topic: DispatchTopic | None = None,
    ) -> PendingAtom:
        """提交 UPDATE（``memory.update``）：授权后解析可读正式基础原子，登记成功才失效基础缓存。

        ``thread`` 与 ``dispatch_topic`` 恰有其一，见 :meth:`_dispatch_topic`。
        """
        scope = self._authorize(actor, WorkspaceOperation.MEMORY_UPDATE)
        result = (await self._reader.resolve_references([request.base_alias], scope=scope))[0]
        if result.kind == "pending":
            raise PendingUpdateNotAllowedError(details={"alias": request.base_alias})
        if result.kind != "atom" or result.atom is None:
            raise ResourceNotFoundError(details={"alias": request.base_alias})
        atom = result.atom
        topic_id = await self._dispatch_topic(thread, dispatch_topic)
        process_id, thread_id = self._resolve_thread(thread)
        pending = self._intents.register_update(
            UpdateFocus(
                base_alias=atom.index.alias or request.base_alias,
                base_uuid=str(atom.id),
                base_version=atom.meta.version,
                instruction=request.instruction,
                content=request.content or None,
            ),
            belong_to=scope.workspace_identity,
            from_actor=scope.actor_identity,
            process_id=process_id,
            thread_id=thread_id,
        )
        self._reader.evict(scope.workspace_identity, atom.id)
        return await self._dispatch(pending, topic_id=topic_id, scope=scope)

    async def _dispatch_topic(
        self, thread: ThreadHandle | None, dispatch_topic: DispatchTopic | None
    ) -> str:
        """取得这一轮的派发话题：线程提交向线程表要，请求级提交调用入口给出的回调。

        二者恰有其一，否则是接线缺陷，以 ``TypeError`` 拒绝。取得派发话题可能要为
        这一轮第一次建立话题，因此排在授权与 UPDATE 的基础冷读之后：被拒的写入
        不会触发路由、建立话题或 LRU 结算。它也在登记的同步段之前，期间线程
        结束或进程收到停止时，同步段照常拒绝登记。
        """
        if thread is not None and dispatch_topic is None:
            return await self._threads.dispatch_topic(thread)
        if thread is None and dispatch_topic is not None:
            return await dispatch_topic()
        raise TypeError("写入与更新需要线程句柄或派发话题的回调，二者恰有其一")

    def _resolve_thread(self, thread: ThreadHandle | None) -> tuple[str | None, str | None]:
        """登记前的同步段：线程提交再解析一次线程，取得进程关联与线程标识。

        请求级提交没有线程，登记不带进程与线程关联。从这里到登记结束不挂起，
        线程结束与停止都不能插入同步段；UPDATE 的基础冷读与取得派发话题都在此
        之前，跨越线程结束或停止时在这里拒绝，不产生意图。
        """
        if thread is None:
            return None, None
        resolved = self._threads.resolve(thread)
        return resolved.process_id, resolved.thread_id

    async def _dispatch(
        self, pending: PendingAtom, *, topic_id: str, scope: IdentityScope
    ) -> PendingAtom:
        """持有派发任务直到提交完成；取消只影响调用方，不撤回已开始的提交。

        等待过程中可能反复收到取消，始终屏蔽到任务结束，避免留下无人持有的
        后台协程。取消优先返回调用方，登记仍按真实接纳结果结算；提交失败的
        根因由提交任务自身记录。
        """
        task = asyncio.create_task(self._submit_intent(pending, topic_id=topic_id, scope=scope))
        await join_shielded(task)
        return pending

    async def _submit_intent(
        self, pending: PendingAtom, *, topic_id: str, scope: IdentityScope
    ) -> None:
        """只经唯一公共意图路由提交，明确拒绝与未知结果使用不同领域错误。

        结局不明时以同一意图重试：提交路由按 ``intent_id`` 幂等，原意图其实已被接纳
        时重试命中原任务，不产生第二个任务。两类错误交给 Agent 时都不携带内部 cause，
        因此根因只在这里记录日志；重试后仍然不明的句柄停在 MATERIALIZING，直到保留期
        满被删除。
        """
        details = {"intent_id": pending.intent_id, "pending_alias": pending.pending_alias}
        try:
            intent = self._memory_intent(pending, topic_id=topic_id)
        except ValueError as error:
            # 本地载荷校验失败确定没有入队，不能留下永远等不到结算的句柄。
            logger.warning(
                "Intent dispatch rejected by local validation: "
                "intent_id=%s, pending_alias=%s, error=%s",
                pending.intent_id,
                pending.pending_alias,
                error,
            )
            self._intents.mark_failed(pending.pending_alias, intent_id=pending.intent_id)
            raise IntentDispatchRejectedError(details=details) from error
        status = await self._request_admission(intent, pending=pending, scope=scope)
        for delay in _DISPATCH_RETRY_DELAYS:
            if status is not MemoryGenerationAdmissionStatus.UNKNOWN:
                break
            await asyncio.sleep(delay)
            status = await self._request_admission(intent, pending=pending, scope=scope)

        if status is MemoryGenerationAdmissionStatus.ACCEPTED:
            return
        if status is MemoryGenerationAdmissionStatus.REJECTED:
            self._intents.mark_failed(pending.pending_alias, intent_id=pending.intent_id)
            raise IntentDispatchRejectedError(details=details)
        logger.error(
            "Intent dispatch outcome unknown after retries: reason=dispatch_unknown, "
            "intent_id=%s, pending_alias=%s, retries=%d",
            pending.intent_id,
            pending.pending_alias,
            len(_DISPATCH_RETRY_DELAYS),
        )
        raise IntentDispatchUnknownError(details=details)

    async def _request_admission(
        self, intent: MemoryIntent, *, pending: PendingAtom, scope: IdentityScope
    ) -> MemoryGenerationAdmissionStatus:
        """提交一次并返回接纳结果；总线异常同样按结果不明返回，根因在这里记日志。"""
        try:
            result = await self._global_bus.request(
                PatchouliRoutes.MEMORY_INTENT_SUBMIT, intent=intent, identity_scope=scope
            )
        except Exception:
            logger.warning(
                "Intent dispatch outcome unknown: intent_id=%s, pending_alias=%s",
                pending.intent_id,
                pending.pending_alias,
                exc_info=True,
            )
            return MemoryGenerationAdmissionStatus.UNKNOWN
        if result.status is MemoryGenerationAdmissionStatus.REJECTED:
            # 拒绝原因由生成侧接纳处记录，这里只关联到登记句柄。
            logger.debug(
                "Intent dispatch rejected by generation admission: "
                "intent_id=%s, pending_alias=%s, detail=%s",
                pending.intent_id,
                pending.pending_alias,
                result.detail,
            )
        elif result.status is MemoryGenerationAdmissionStatus.UNKNOWN:
            logger.warning(
                "Intent dispatch admission unknown: intent_id=%s, pending_alias=%s, detail=%s",
                pending.intent_id,
                pending.pending_alias,
                result.detail,
            )
        return result.status

    @staticmethod
    def _memory_intent(pending: PendingAtom, *, topic_id: str) -> MemoryIntent:
        """逐字段转换为公共意图，不依赖 focus 与意图字段一一同名。

        focus 新增字段只需要在这里补映射，不会在登记之后以 ``TypeError`` 绕过
        确定性拒绝、把句柄留在 MATERIALIZING。
        """
        focus = pending.focus
        if isinstance(focus, WriteFocus):
            return MemoryIntent(
                kind="write",
                topic_id=topic_id,
                intent_id=pending.intent_id,
                pending_alias=pending.pending_alias,
                content=focus.content,
                title=focus.title,
                reason=focus.reason,
            )
        return MemoryIntent(
            kind="update",
            topic_id=topic_id,
            intent_id=pending.intent_id,
            pending_alias=pending.pending_alias,
            instruction=focus.instruction,
            content=focus.content,
            base_alias=focus.base_alias,
            base_uuid=focus.base_uuid,
            base_version=focus.base_version,
        )

    async def resolve_references(
        self,
        request: ResolveReferencesRequest,
        *,
        actor: WorkingActor,
    ) -> list[ReferenceResolution]:
        """读取引用（``memory.read``），为本次交付的正式原子逐一记录引用。

        逐项结果保留顺序与重复项，引用记录在本请求内按 memory id 去重。
        只处理 atom 或带可读原子的 redirect；缓存命中也记录。记录失败只
        影响日志，不改变读取结果，取消则继续传播给调用方。
        """
        scope = self._authorize(actor, WorkspaceOperation.MEMORY_READ)
        return await self._resolve_and_record_citations(list(request.aliases), scope=scope)

    async def resolve_run_target(
        self,
        request: ResolveRunTargetRequest,
        *,
        actor: WorkingActor,
    ) -> ReferenceResolution:
        """解析 RUN 的单个执行目标（``memory.run``），只为可执行的目标记录正式引用。

        RUN 与 READ 独立授权；资源可见性沿用同一读取视图，返回中立解析结果，
        目标是否可执行仍由 Actor 的工具运行时判断并拒绝。只有 ``CODE_SNIPPET``
        能被 RUN 执行，其他类型的目标会被拒绝、不构成一次使用，因此不记录引用。
        """
        scope = self._authorize(actor, WorkspaceOperation.MEMORY_RUN)
        resolutions = await self._resolve_and_record_citations(
            [request.alias], scope=scope, runnable_only=True
        )
        return resolutions[0]

    async def _resolve_and_record_citations(
        self,
        aliases: list[str],
        *,
        scope: IdentityScope,
        runnable_only: bool = False,
    ) -> list[ReferenceResolution]:
        """复用 READ/RUN 的解析与引用记录，权限只由各自的公共入口检查。

        ``runnable_only`` 为真时（RUN）只为 ``CODE_SNIPPET`` 原子记录引用。
        """
        results = await self._reader.resolve_references(aliases, scope=scope)
        cited: set[UUID] = set()
        for result in results:
            atom = result.atom
            if result.kind not in ("atom", "redirect") or atom is None or atom.id in cited:
                continue
            if runnable_only and atom.index.memory_type != MemoryType.CODE_SNIPPET:
                continue
            # 去重发生在尝试前：同批失败不重试，以免已提交但应答失败时重复计数。
            cited.add(atom.id)
            try:
                await self._global_bus.request(
                    PatchouliRoutes.RECORD_MEMORY_CITATION,
                    memory_id=atom.id,
                    identity_scope=scope,
                    source="workspace.reference_read",
                )
            except Exception:
                # 生命周期反馈为 best-effort；CancelledError 不属于 Exception，
                # 必须正常传播，不能把已经取消的读取变为成功。
                logger.warning(
                    "Failed to record memory citation for memory_id=%s source=%s",
                    atom.id,
                    "workspace.reference_read",
                    exc_info=True,
                )
        return results

    async def retrieve(
        self,
        request: RetrieveRequest,
        *,
        actor: WorkingActor,
    ) -> list[MemoryAtom]:
        """语义检索（``memory.search``）：保持领域排序，结果协作预热缓存，并为结果记录命中。

        调用方只提交检索参数：检索请求由本层用授权返回的可信 scope 构造，
        ``IdentityScope`` 不进入调用方输入，
        不能据请求体重新选择 Workspace。
        """
        scope = self._authorize(actor, WorkspaceOperation.MEMORY_SEARCH)
        retrieval = RetrievalRequest(
            semantic_query=request.semantic_query,
            keywords=list(request.keywords),
            identity_scope=scope,
            filters=request.filters,
        )
        memories = await self._reader.search(retrieval, scope=scope)
        await self._record_retrieval_hits(memories, scope=scope)
        return memories

    async def _record_retrieval_hits(
        self, memories: list[MemoryAtom], *, scope: IdentityScope
    ) -> None:
        """检索即为交付的记忆记录命中，不论调用方之后是否用上。

        任务进程的预检索、Agent 的检索与外部 harness 的检索都经这里，命中因此
        与检索同时发生。命中是生命周期的强化信号，尽力而为：记录失败只记日志，
        不改变检索结果；取消照常传播。同一批的去重由 Patchouli 完成。
        """
        if not memories:
            return
        try:
            await self._global_bus.request(
                PatchouliRoutes.RECORD_RETRIEVAL_HITS,
                tuple(memory.id for memory in memories),
                identity_scope=scope,
                source=_SEARCH_HIT_SOURCE,
            )
        except Exception:
            logger.warning(
                "Failed to record retrieval hits: count=%s source=%s",
                len(memories),
                _SEARCH_HIT_SOURCE,
                exc_info=True,
            )

    def _authorize(self, actor: WorkingActor, operation: WorkspaceOperation) -> IdentityScope:
        """在 backing 调用前对驻留 Workspace 执行操作授权，返回组装的可信 scope。"""
        return self._authorizer.authorize_agent(actor, operation, actor.workspace)
