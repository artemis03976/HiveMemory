"""workspace 单一操作入口：兑现凭据并薄分派到对应的能力方法。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from hivememory.core.models.agent import AgentProfile
from hivememory.core.models.memory import MemoryAtom
from hivememory.core.models.pending import PendingAtom
from hivememory.core.models.reference import ReferenceResolution
from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.memory import MemoryApplicationService
from hivememory.workspace.contracts.operations import (
    CancelIntentsRequest,
    ExecutionCredential,
    ExecutionCredentialRevokedError,
    GetAgentProfileRequest,
    OperationRequest,
    ResolveReferencesRequest,
    RetrieveRequest,
    SubmitUpdateIntentRequest,
    SubmitWriteIntentRequest,
)
from hivememory.workspace.credentials import ExecutionBinding, ExecutionCredentialRegistry
from hivememory.workspace.intents.registry import WriteIntentRegistry

# 产生写入意图登记的请求：返回入口后都要复查凭据，见 ``execute``。
_INTENT_SUBMISSIONS = (SubmitWriteIntentRequest, SubmitUpdateIntentRequest)


class WorkspaceOperationEntry:
    """无业务状态的入口，身份与进程关联只取自凭据表。"""

    def __init__(
        self,
        memory: MemoryApplicationService,
        *,
        agent: AgentApplicationService,
        credential_registry: ExecutionCredentialRegistry,
        intent_registry: WriteIntentRegistry,
    ) -> None:
        self._memory = memory
        self._agent = agent
        self._credentials = credential_registry
        self._intents = intent_registry
        # 泛型结果由公开契约约束；异构分派表仅在入口内部擦除类型。
        self._handlers: dict[
            type[OperationRequest[Any]], Callable[[Any, ExecutionBinding], Awaitable[Any]]
        ] = {
            SubmitWriteIntentRequest: self._submit_write,
            SubmitUpdateIntentRequest: self._submit_update,
            CancelIntentsRequest: self._cancel_intents,
            ResolveReferencesRequest: self._resolve_references,
            RetrieveRequest: self._retrieve,
            GetAgentProfileRequest: self._get_agent_profile,
        }

    async def execute[R](
        self, request: OperationRequest[R], *, credential: ExecutionCredential
    ) -> R:
        """分派前兑现凭据；意图提交在返回后的同步段补偿关闭期间的登记。

        WRITE 与 UPDATE 返回入口后都复查凭据：能力方法在登记前只要有一次
        await（当前是 UPDATE 的基础冷读），进程就可能已经关闭并取消了本进程
        的 PENDING 意图。统一复查使这条保证不依赖能力方法内部是否等待；只读
        请求在吊销时已在途的仍正常完成。新增会登记意图、需随进程关闭撤回的请求
        时同样加入。
        """
        binding = self._credentials.resolve(credential)
        handler = self._handlers.get(type(request))
        if handler is None:
            raise TypeError(f"Unsupported operation request: {type(request).__name__}")
        result = await handler(request, binding)
        if isinstance(request, _INTENT_SUBMISSIONS):
            try:
                self._credentials.resolve(credential)
            except ExecutionCredentialRevokedError:
                # 能力方法返回至这里没有挂起点。补偿直接撤回登记，不能在
                # context 已失效后重新授权，也不能 await 后才清理游离意图。
                pending = cast(PendingAtom, result)
                self._intents.cancel_aliases([pending.pending_alias], process_id=binding.process_id)
                raise
        return cast(R, result)

    async def _submit_write(
        self, request: SubmitWriteIntentRequest, binding: ExecutionBinding
    ) -> PendingAtom:
        """WRITE 的进程关联与目标由入口绑定，能力层仍逐次授权。"""
        return await self._memory.submit_write_intent(
            request.focus,
            process_id=binding.process_id,
            target_workspace=binding.target_workspace,
            access=binding.access,
        )

    async def _submit_update(
        self, request: SubmitUpdateIntentRequest, binding: ExecutionBinding
    ) -> PendingAtom:
        """基础引用解析与缓存失效保持在 UPDATE 能力方法内。"""
        return await self._memory.submit_update_intent(
            request.base_alias,
            request.instruction,
            request.content,
            process_id=binding.process_id,
            target_workspace=binding.target_workspace,
            access=binding.access,
        )

    async def _cancel_intents(
        self, request: CancelIntentsRequest, binding: ExecutionBinding
    ) -> list[str]:
        """只撤回绑定进程仍为 PENDING 的指定意图。"""
        return await self._memory.cancel_intents(
            list(request.aliases),
            process_id=binding.process_id,
            target_workspace=binding.target_workspace,
            access=binding.access,
        )

    async def _resolve_references(
        self, request: ResolveReferencesRequest, binding: ExecutionBinding
    ) -> list[ReferenceResolution]:
        """请求不携带目标，引用解析沿用能力层的逐项结果。"""
        return await self._memory.resolve_references(
            list(request.aliases),
            target_workspace=binding.target_workspace,
            access=binding.access,
        )

    async def _retrieve(
        self, request: RetrieveRequest, binding: ExecutionBinding
    ) -> list[MemoryAtom]:
        """SEARCH 只提交检索参数，授权与缓存预热均由能力层负责。"""
        return await self._memory.retrieve(
            semantic_query=request.semantic_query,
            keywords=list(request.keywords),
            top_k=request.top_k,
            filters=request.filters,
            target_workspace=binding.target_workspace,
            access=binding.access,
        )

    async def _get_agent_profile(
        self, request: GetAgentProfileRequest, binding: ExecutionBinding
    ) -> AgentProfile:
        """CALL 图纸读取沿用主线程绑定，执行权限仍由 Alice 的 Profile 控制。"""
        return await self._agent.get_agent_profile(
            request.agent_alias,
            target_workspace=binding.target_workspace,
            access=binding.access,
        )


__all__ = ["WorkspaceOperationEntry"]
