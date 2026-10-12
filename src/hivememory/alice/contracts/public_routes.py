"""通过 GlobalSystemBus 暴露的 Alice 执行路由。"""

# route 声明在类体中生成常量，不是实例方法。
# ruff: noqa: N802, N805

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.contracts.route import route

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from typing import Any

    from hivememory.workspace.contracts import (
        CPUExecutionResult,
        CPUInputManifest,
        OperationSubmitter,
    )


class AliceRoutes:
    """两种输出形态各有确定的返回类型，共用同一套执行骨架。"""

    @route("alice.public.run_agent")
    async def RUN_AGENT(
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        submit_operation: OperationSubmitter,
    ) -> CPUExecutionResult:
        """等待执行终态并返回 CPU 中立结果。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    @route("alice.public.run_agent_stream")
    async def RUN_AGENT_STREAM(
        input_manifest: CPUInputManifest,
        generation_options: dict[str, Any] | None = None,
        *,
        submit_operation: OperationSubmitter,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """返回最后一项为 done 的交互流，关闭流时收尾对应 runner。"""
        raise NotImplementedError  # 仅声明签名，不执行此函数。

    ALL = (RUN_AGENT, RUN_AGENT_STREAM)


__all__ = ["AliceRoutes"]
