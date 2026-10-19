"""Patchouli 记忆意图提交 application API（公开路由 ``patchouli.public.memory_intent.submit``）。

承接中立的记忆意图参数并转换为内部生成任务：由 controller admission 与
generation engine 决定生成、更新、合并或丢弃；**不映射为管理 CRUD，也
不由调用方构造内部任务投影**。本层只接收授权点传入的 ``IdentityScope``。

意图的 ``intent_id`` 与 ``pending_alias`` 由 workspace 登记产生并原样使用：
结算事件（FAILED、CANCELLED 只携带别名）因此回到登记的同一句柄。
``intent_id`` 是生成任务的幂等键：同一 intent 携带相同载荷重试命中 controller
幂等复用并返回原任务；载荷变化不会产生第二个任务（at-most-one-canonical）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from hivememory.core.models import (
    ActorIdentity,
    IdentityScope,
    WorkspaceIdentity,
    require_identity_scope,
)
from hivememory.core.models.pending import (
    PendingAtomMaterializeTask,
    UpdateFocus,
    WriteFocus,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.memory_intent import MemoryIntent, MemoryIntentSubmissionResult
from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmissionStatus

if TYPE_CHECKING:
    from hivememory.patchouli.runtime.bus import PatchouliBus


class MemoryIntentSubmissionService:
    """经 Patchouli 生成提交链的公开意图提交 API。"""

    def __init__(self, *, bus: PatchouliBus) -> None:
        self._bus = bus

    async def submit_memory_intent(
        self,
        *,
        intent: MemoryIntent,
        identity_scope: IdentityScope,
    ) -> MemoryIntentSubmissionResult:
        """提交记忆意图，返回生成队列的接纳结果；应用进度经任务观察 API 查询。"""
        scope = require_identity_scope(identity_scope)

        task = self._build_materialize_task(intent, scope.workspace_identity, scope.actor_identity)
        admission = await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            task=task,
            topic_id=intent.topic_id,
            belong_to=scope.workspace_identity,
        )
        if admission.task is None:
            detail = (
                "generation admission rejected the intent"
                if admission.status is MemoryGenerationAdmissionStatus.REJECTED
                else "generation admission outcome unknown"
            )
            return MemoryIntentSubmissionResult(status=admission.status, detail=detail)
        return MemoryIntentSubmissionResult(
            status=admission.status,
            task_id=admission.task.task_id,
            task_status=admission.task.status.value,
        )

    # ---- 内部辅助 ----

    @staticmethod
    def _build_materialize_task(
        intent: MemoryIntent,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
    ) -> PendingAtomMaterializeTask:
        """把中立意图转换为内部生成任务投影（本 API 的内部实现细节）。"""
        focus: WriteFocus | UpdateFocus
        source_verb: Literal["WRITE", "UPDATE"]
        if intent.kind == "write":
            focus = WriteFocus(
                content=intent.content or "",
                reason=intent.reason,
                title=intent.title,
            )
            source_verb = "WRITE"
        else:
            focus = UpdateFocus(
                instruction=intent.instruction or "",
                content=intent.content,
                base_uuid=intent.base_uuid or "",
                base_alias=intent.base_alias or "",
            )
            source_verb = "UPDATE"
        return PendingAtomMaterializeTask(
            pending_alias=intent.pending_alias,
            intent_id=intent.intent_id,
            source_verb=source_verb,
            belong_to=belong_to,
            from_actor=from_actor,
            focus=focus,
        )


__all__ = [
    "MemoryIntentSubmissionService",
]
