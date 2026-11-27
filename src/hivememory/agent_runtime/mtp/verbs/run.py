"""RUN 指令的语义处理。"""

from __future__ import annotations

from hivememory.agent_runtime.mtp.context import MTPExecutionContext
from hivememory.agent_runtime.tools.types import Syscall
from hivememory.core.mtp import (
    MTPCommand,
    MTPResponse,
    MTPResponseStatus,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
)
from hivememory.workspace.contracts import (
    ResolveRunTargetRequest,
)


async def handle_run(
    command: MTPCommand,
    context: MTPExecutionContext,
    *,
    syscall_registry: dict[str, Syscall],
) -> MTPResponse:
    """
    处理 RUN 指令

    两层分发:
    - 系统工具: 命中 syscall 注册表的 sys_* 工具在本地执行
    - 记忆别名: 经 ResolveRunTargetRequest 授权并解析，解析到的记忆一律以“不可运行”
      拒绝：能否运行是某一份内容相对某个环境的性质，记忆上没有这项判定，生成链
      写下的也只是知识，不产出可执行物；解析到的话题同样拒绝，话题只能 READ

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

    # 记忆别名。RUN 以自己的操作请求进入读取视图，与 READ 的许可独立，
    # 没有 memory.run 的执行者在这里被拒绝。
    # StorageOfflineError / BusRouteUnavailableError 会直接传播到 _route_and_execute
    resolved = await context.submit_operation(ResolveRunTargetRequest(alias=alias))
    if resolved.kind == "topic":
        raise InvalidArgumentError(
            message_key="mtp.run.topic_not_runnable",
            params={"alias": alias},
        )
    if resolved.kind == "pending":
        raise InvalidArgumentError(
            message_key="mtp.run.pending_not_runnable",
            params={"alias": alias},
        )
    if resolved.kind == "redirect" and resolved.atom is not None:
        alias = resolved.canonical_alias or resolved.atom.get_alias()
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
    raise InvalidArgumentError(
        message_key="mtp.run.memory_not_runnable",
        params={"alias": alias},
    )
