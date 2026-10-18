"""RUN 指令的语义处理。"""

from __future__ import annotations

import logging

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.tools.types import Syscall
from hivememory.agent_runtime.tools.user import execute_user_tool
from hivememory.core.models import MemoryType
from hivememory.core.mtp import (
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
    MTPWarningInfo,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.workspace.contracts import (
    ResolveRunTargetRequest,
)

logger = logging.getLogger(__name__)


async def handle_run(
    command: MTPCommand,
    context: MTPExecutionContext,
    *,
    syscall_registry: dict[str, Syscall],
    timeout_seconds: int,
) -> MTPResponse:
    """
    处理 RUN 指令

    两层分发:
    - 系统工具快速路径: 命中 syscall 注册表的 sys_* 工具在本地执行
    - 用户工具路径: 经 ResolveRunTargetRequest 解析 CODE_SNIPPET 记忆 → 沙箱执行

    Type B 动作类响应

    Args:
        command: RUN 指令 (target=tool_alias, args=key-value pairs)

    Returns:
        MTPResponse: 执行状态描述
    """
    alias = command.target.single_alias
    if alias is None:
        raise InvalidArgumentError(message_key="mtp.run.missing_single_target")

    syscall = syscall_registry.get(alias)
    if syscall is None and alias.startswith("sys_"):
        raise AliasNotFoundError(
            message_key="mtp.run.tool_not_found",
            params={"alias": alias},
        )
    if syscall is not None:
        # 系统工具不设 Agent 权限，在 Alice 本地执行。
        result = syscall.handler(command.args)
        return MTPResponse(status=MTPResponseStatus.SUCCESS, content=result.content)

    # 用户工具路径。RUN 以自己的操作请求进入读取视图，
    # 与 READ 的许可独立；交付执行目标时仍由 workspace 记录正式引用。
    # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
    resolved = await context.submit_operation(ResolveRunTargetRequest(alias=alias))
    warnings: list[MTPWarningInfo] = []
    if resolved.kind == "pending":
        raise InvalidArgumentError(
            message_key="mtp.run.pending_not_runnable",
            params={"alias": alias},
        )
    if resolved.kind == "redirect" and resolved.atom is not None:
        canonical_alias = resolved.canonical_alias or resolved.atom.get_alias()
        warnings.append(
            MTPWarningInfo(
                message_key="mtp.run.alias_redirected",
                params={
                    "requested_alias": alias,
                    "canonical_alias": canonical_alias,
                },
            )
        )
        alias = canonical_alias
    elif resolved.kind in {"discarded", "failed", "expired"}:
        raise AliasNotFoundError(
            message_key="mtp.run.terminal_alias_not_runnable",
            params={"alias": alias, "status": resolved.kind},
        )
    if resolved.kind not in {"atom", "redirect"} or resolved.atom is None:
        raise AliasNotFoundError(
            message_key="mtp.run.alias_not_found",
            params={"alias": alias},
        )
    atom = resolved.atom

    # 校验类型必须是 CODE_SNIPPET
    if atom.index.memory_type != MemoryType.CODE_SNIPPET:
        raise MemoryTypeMismatchError(
            message_key="mtp.run.type_mismatch",
            params={"alias": alias, "memory_type": atom.index.memory_type.value},
        )

    # 使用缓存的代码执行
    code = atom.payload.content
    logger.info(f"User tool executing: alias='{alias}', UUID={atom.id}")
    response = execute_user_tool(code, command.args, timeout_seconds=timeout_seconds)
    if warnings:
        response.warnings = [*warnings, *response.warnings]
    return response
