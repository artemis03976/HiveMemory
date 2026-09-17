"""Patchouli 领域提交/结果端口适配器（``DomainMutationPort``/``DomainResultPort``）。

提交语义（父计划 5.7 节与 WRX-0 冻结结论）：
- ``apply_interaction`` 经 ``InteractionSubmissionQueue`` 提交，由 Patchouli
  感知层应用；队列接纳即回执，应用进度按 interaction_id 跟踪；
- ``submit_memory_intent`` 经 Patchouli 总线请求生成提交链
  （``GENERATION_SUBMIT_ACTIVE``），由 controller admission 与 generation
  engine 决定生成、更新、合并或拒绝；**不得旁路到管理服务或直接 upsert**。
  ``pending_alias`` 由 ``intent_id`` 确定性派生：同一 intent_id 携带相同
  载荷重试时，controller 按 spec 相等做幂等复用并返回原任务；载荷不同
  则按 intent 冲突确定性拒绝（at-most-one-canonical 语义）；
- ``get_submission_result`` 按任务归属投影（scope + submitted_by）做
  越权判定：跨 scope、无归属与不存在的任务统一返回 not found，
  不向查询方泄漏任务存在性。

适配器只做协议与坐标转换：``MemoryIntentRequest`` 是语义等价的领域参数，
内部桥接为既有跨边界投影 ``PendingAtomMaterializeTask``（该类型保持为
生成链入口形状，不作为端口参数类型外泄；桥接产生的 alias 背后没有
Alice PendingAtom 注册，settlement/failed 事件只作为领域结果投影）。
"""

from __future__ import annotations

import hashlib
import logging
import re
from uuid import uuid4

from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models.pending import (
    PendingAtomMaterializeTask,
    UpdateFocus,
    WriteFocus,
)
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.control.interaction_submission import (
    InteractionSubmission,
    InteractionSubmissionQueue,
)
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceOperation,
    require_access_context,
)
from hivememory.workspace.ports import DomainMutationPort, DomainResultPort
from hivememory.workspace.projections import (
    DomainHandle,
    DomainResult,
    DomainSubmission,
    InteractionApplyRequest,
    InteractionApplyResult,
    MemoryIntentRequest,
)

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 30) -> str:
    """将文本转为 alias 友好的 slug 片段（与 PendingAtom alias 风格一致）。"""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s_]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len].rstrip("_")


def _intent_token(intent_id: str) -> str:
    """从 intent_id 派生确定性短 token，保证重试生成同一 pending_alias。"""
    return hashlib.sha1(intent_id.encode("utf-8")).hexdigest()[:4]


class PatchouliDomainGateway(DomainMutationPort, DomainResultPort):
    """经 Patchouli 总线与交互队列实现领域提交/结果端口。

    依赖注入的都是 Patchouli 既有入口（局部总线 + 交互队列），不直接
    持有 canonical store、generation 队列内部状态或任何 Workspace cache。
    """

    def __init__(
        self,
        bus: PatchouliBus,
        interaction_queue: InteractionSubmissionQueue,
    ) -> None:
        self._bus = bus
        self._interaction_queue = interaction_queue

    # ---- DomainMutationPort ----

    async def apply_interaction(self, request: InteractionApplyRequest) -> InteractionApplyResult:
        """提交已发生的交互载荷；队列幂等键为 interaction_id。"""
        access = require_access_context(
            request.access, operation=WorkspaceOperation.INTERACTION_SUBMIT
        )
        payload = request.payload
        if payload is None:
            raise ValueError("InteractionApplyRequest.payload 不能为空")

        interaction_id = request.interaction_id or f"interaction_{uuid4().hex[:12]}"
        topic_id = request.requested_topic_id
        submission = InteractionSubmission(
            identity_scope=access.identity_scope,
            interaction_id=interaction_id,
            payload=payload,
            requested_topic_id=topic_id,
            ordering_key=f"topic:{topic_id}",
            origin="workspace_port",
            correlation={
                "principal_id": access.principal.principal_id,
                "submitted_by": access.identity_scope.actor_identity.agent_id,
            },
        )
        receipt = await self._interaction_queue.submit(submission)
        return InteractionApplyResult(
            interaction_id=receipt.interaction_id,
            work_id=receipt.work_id,
            state=receipt.state.value,
        )

    async def submit_memory_intent(self, request: MemoryIntentRequest) -> DomainSubmission:
        """提交记忆意图；结果由 Patchouli 生成链决定并经结果端口查询。

        同一 ``intent_id`` 与相同载荷的重试经确定性 ``pending_alias`` 命中
        controller 幂等复用，返回原任务句柄；载荷变化或确定性拒绝返回
        ``accepted=False``。
        """
        access = require_access_context(
            request.access, operation=WorkspaceOperation.MEMORY_INTENT_SUBMIT
        )
        task = self._build_materialize_task(request, access)
        accepted = await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_ACTIVE,
            tasks=[task],
            topic_id=request.topic_id,
            identity_scope=access.identity_scope,
            submitted_by=access.principal.principal_id,
        )
        tasks = list(accepted or [])
        if not tasks:
            # 空响应表示确定性拒绝（intent 冲突 / spec 构建失败，settler 已
            # 发布失败投影）或批量结果缺失；两种情况都不能声称"已接纳"。
            return DomainSubmission(
                accepted=False,
                detail="generation admission did not accept the intent",
            )
        admitted = tasks[0]
        return DomainSubmission(
            accepted=True,
            handle=DomainHandle(task_id=admitted.task_id),
            task_status=admitted.status.value,
        )

    # ---- DomainResultPort ----

    async def get_submission_result(
        self,
        access: WorkspaceAccessContext,
        handle: DomainHandle,
    ) -> DomainResult:
        """查询提交结果投影；跨 scope、无归属与不存在统一按 not found 拒绝。

        归属校验（父计划 5.6.4）：任务必须携带 Workspace scope 且与查询方
        一致；缺失归属（legacy 快照）一律 fail closed。三种失败共用同一
        错误，避免向查询方泄漏其他 Workspace 的任务存在性。
        """
        access = require_access_context(access, operation=WorkspaceOperation.TASK_OBSERVE)
        task = await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_GET,
            handle.task_id,
        )
        task_scope = getattr(task, "identity_scope", None) if task is not None else None
        if task is None or task_scope != access.identity_scope:
            raise ResourceNotFoundError(details={"task_id": handle.task_id})

        return DomainResult(
            task_id=task.task_id,
            status=task.status.value,
            canonical_alias=task.canonical_alias,
            error=task.error,
            topic_id=task.topic_id,
            pending_alias=task.pending_alias,
            identity_scope=task.identity_scope,
            submitted_by=task.submitted_by,
        )

    # ---- 内部辅助 ----

    @staticmethod
    def _build_materialize_task(
        request: MemoryIntentRequest,
        access: WorkspaceAccessContext,
    ) -> PendingAtomMaterializeTask:
        """把语义意图请求桥接为生成链的跨边界物化投影。

        ``pending_alias`` 由 ``intent_id`` 确定性派生（slug + intent token），
        使同一 intent 的重试构造出逐字段相等的任务，命中 controller 的
        幂等复用；不同载荷复用同一 intent_id 则触发 intent 冲突拒绝。
        """
        intent_id = request.intent_id or f"intent_{uuid4().hex[:12]}"
        token = _intent_token(intent_id)
        if request.kind == "write":
            slug = _slugify(request.title or request.content or "") or "untitled"
            pending_alias = f"draft_{slug}_{token}"
            focus = WriteFocus(
                content=request.content or "",
                reason=request.reason,
                title=request.title,
            )
            source_verb = "WRITE"
        else:
            slug = _slugify(request.base_alias or "") or "base"
            pending_alias = f"rev_{slug}_{token}"
            focus = UpdateFocus(
                instruction=request.instruction or "",
                content=request.content,
                base_uuid=request.base_uuid or "",
                base_alias=request.base_alias or "",
            )
            source_verb = "UPDATE"
        return PendingAtomMaterializeTask(
            pending_alias=pending_alias,
            intent_id=intent_id,
            source_verb=source_verb,
            identity_scope=access.identity_scope,
            focus=focus,
        )


__all__ = ["PatchouliDomainGateway"]
