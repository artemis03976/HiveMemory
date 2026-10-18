from __future__ import annotations

import logging
from dataclasses import dataclass

from hivememory.agent_runtime import ExecutionFrame
from hivememory.core.errors import ThreadContextRevokedError
from hivememory.core.mtp import MTPCallRequest
from hivememory.engines.memory_compiler import (
    MemoryCompileOptions,
    MemoryCompiler,
    MemoryEnvelopeTarget,
)
from hivememory.workspace.contracts import (
    Callee,
    DeriveCallThreadRequest,
    ResolveReferencesRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CallContext:
    """CALL 的编排上下文：被调用方与调用方共享的上下文，不包含 frame 或 CALL ledger 状态。"""

    callee: Callee
    shared_context: str = ""


class CallContextProvider:
    """解析 CALL target 与 context_refs，供 CallCoordinator 组装 callee frame。"""

    async def provide(
        self,
        caller_frame: ExecutionFrame,
        request: MTPCallRequest,
    ) -> CallContext:
        """经调用方提交函数派生子线程，再以调用方权限编译共享上下文。

        内置 Profile 同样提交 CALL 目标请求，不能绕过 ``agent.call`` 授权；
        context_refs 另提交 READ 请求，仍需要 ``memory.read``。错误交给 CALL 响应边界统一呈现，
        此处不持有另一份 Profile 缓存。
        """
        callee = await caller_frame.submit_operation.derive_call_thread(
            DeriveCallThreadRequest(agent_alias=request.target_alias)
        )
        shared_context = await self._resolve_shared_context(
            aliases=request.context_refs,
            caller_frame=caller_frame,
        )
        return CallContext(callee=callee, shared_context=shared_context)

    async def _resolve_shared_context(
        self,
        *,
        aliases: list[str],
        caller_frame: ExecutionFrame,
    ) -> str:
        """逐项解析 context_refs，并编译为子 Agent 的共享上下文。

        每个引用单独提交 READ 请求：某个引用冷读失败（如存储不可达）时只
        跳过该引用，已命中缓存的其余引用照常交付；合并为一次批量请求会让
        单次失败丢弃全部共享上下文。
        """
        if not aliases:
            return ""

        compiler = MemoryCompiler()
        sources = []
        for alias in aliases:
            try:
                resolved = (
                    await caller_frame.submit_operation(ResolveReferencesRequest((alias,)))
                )[0]
            except ThreadContextRevokedError:
                # 线程结束不是单条引用缺失，必须交给 CALL 出口提示停止操作。
                raise
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
            MemoryCompileOptions(language=caller_frame.language),
        ).text


__all__ = ["CallContext", "CallContextProvider"]
