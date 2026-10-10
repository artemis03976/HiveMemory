from __future__ import annotations

import logging
from dataclasses import dataclass

from hivememory.agent_runtime.models import ExecutionFrame
from hivememory.core.errors import OperationDeniedError, ResourceUnavailableError
from hivememory.core.models import AgentProfile
from hivememory.core.mtp import MTPCallRequest
from hivememory.core.mtp.exceptions import (
    BusRouteUnavailableError,
    MTPError,
    PermissionDeniedError,
    SystemFault,
)
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.workspace.contracts import GetAgentProfileRequest, ResolveReferencesRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CallContext:
    """CALL target 的编排上下文，不包含 frame 或 CALL ledger 状态。"""

    agent_profile: AgentProfile
    shared_context: str = ""


class CallContextProvider:
    """解析 CALL target 与 context_refs，供 CallCoordinator 组装 callee frame。"""

    async def provide(
        self,
        caller_frame: ExecutionFrame,
        request: MTPCallRequest,
    ) -> CallContext:
        """经调用方提交函数读取目标 Profile，再编译受控共享上下文。

        内置 Profile 同样提交读取请求，不能绕过 workspace 的 ``profile.read``
        授权；Alice 只保留 MTP 错误映射，不持有另一份 Profile 缓存。
        """
        if caller_frame.submit_operation is None:
            raise RuntimeError("CALL 缺少操作提交函数")
        try:
            profile = await caller_frame.submit_operation(
                GetAgentProfileRequest(agent_alias=request.target_alias)
            )
        except OperationDeniedError as error:
            raise PermissionDeniedError(
                message_key="mtp.permission.verb_denied",
                params={"verb": "CALL"},
                cause=error,
            ) from error
        except ResourceUnavailableError as error:
            raise BusRouteUnavailableError(cause=error) from error
        except MTPError:
            raise
        except Exception as error:
            raise SystemFault(
                message_key="mtp.call.profile_load_failed",
                params={"agent_alias": request.target_alias},
                cause=error,
            ) from error
        shared_context = await self._resolve_shared_context(
            aliases=request.context_refs,
            caller_frame=caller_frame,
        )
        return CallContext(
            agent_profile=profile,
            shared_context=shared_context,
        )

    async def _resolve_shared_context(
        self,
        *,
        aliases: list[str],
        caller_frame: ExecutionFrame,
    ) -> str:
        """逐项解析 context_refs，并编译为子 Agent 的共享上下文。"""
        if not aliases:
            return ""

        compiler = MemoryCompiler()
        sources = []
        if caller_frame.submit_operation is None:
            raise RuntimeError("CALL context_refs 缺少操作提交函数")
        for alias in aliases:
            try:
                resolved = (
                    await caller_frame.submit_operation(ResolveReferencesRequest((alias,)))
                )[0]
            except Exception as error:
                logger.warning("Failed to resolve context_ref %s: %s", alias, error)
                continue
            # redirect 必须带有调用方可读的正式原子；不可读目标不交给编译器。
            if (resolved.kind == "pending" and resolved.pending is not None) or (
                resolved.kind in {"redirect", "atom"} and resolved.atom is not None
            ):
                sources.append(resolved)
            else:
                logger.warning("Context ref alias not found: %s", alias)

        if not sources:
            logger.warning("No rendered context returned for context_refs: %s", aliases)
            return ""

        return compiler.compile(
            sources,
            MemoryEnvelopeTarget.SHARED_CONTEXT_INJECTION,
            MemoryCompileOptions(language=caller_frame.agent_profile.language),
        ).text


__all__ = ["CallContext", "CallContextProvider"]
