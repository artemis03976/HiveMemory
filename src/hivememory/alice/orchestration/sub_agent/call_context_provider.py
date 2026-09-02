from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hivememory.agent_runtime.models import ExecutionFrame, MTPExecutionContext
from hivememory.core.models import AgentProfile
from hivememory.core.mtp import MTPCallRequest
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)

if TYPE_CHECKING:
    from hivememory.agent_runtime.aliases import RuntimeAliasResolver
    from hivememory.alice.runtime.profile_resolver import AgentProfileResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CallContext:
    """CALL target 的编排上下文，不包含 frame 或 CALL ledger 状态。"""

    agent_profile: AgentProfile
    shared_context: str = ""


class CallContextProvider:
    """解析 CALL target 与 context_refs，供 CallCoordinator 组装 callee frame。"""

    def __init__(
        self,
        profile_resolver: AgentProfileResolver,
        alias_resolver: RuntimeAliasResolver,
    ) -> None:
        self._profile_resolver = profile_resolver
        self._alias_resolver = alias_resolver

    async def provide(
        self,
        caller_frame: ExecutionFrame,
        request: MTPCallRequest,
    ) -> CallContext:
        """按 caller identity 解析目标 profile 与受控共享上下文。"""
        profile = await self._profile_resolver.resolve(
            request.target_alias,
            identity_scope=caller_frame.identity_scope,
        )
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
        context = MTPExecutionContext(runtime_scope=caller_frame.runtime_scope)
        for alias in aliases:
            try:
                resolved = await self._alias_resolver.resolve(alias, context=context)
            except Exception as error:
                logger.warning("Failed to resolve context_ref %s: %s", alias, error)
                continue
            if resolved.kind in {"pending", "redirect", "atom"} and (
                resolved.pending is not None or resolved.atom is not None
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
