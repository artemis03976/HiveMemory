"""Alice 把 actor 操作快照翻译为本次执行的 MTP 动词集合。"""

from __future__ import annotations

from hivememory.core.access import WorkspaceOperation
from hivememory.core.mtp.models import MTPVerb

_OPERATION_VERBS = {
    WorkspaceOperation.MEMORY_READ: MTPVerb.READ.value,
    WorkspaceOperation.MEMORY_SEARCH: MTPVerb.SEARCH.value,
    WorkspaceOperation.MEMORY_WRITE: MTPVerb.WRITE.value,
    WorkspaceOperation.MEMORY_UPDATE: MTPVerb.UPDATE.value,
    WorkspaceOperation.AGENT_CALL: MTPVerb.CALL.value,
    WorkspaceOperation.MEMORY_RUN: MTPVerb.RUN.value,
}


def mtp_verbs_for_operations(operations: frozenset[WorkspaceOperation]) -> frozenset[str]:
    """翻译提示词与执行策略；系统工具不设权限，因此 RUN 始终可用。

    没有 ``memory.run`` 时，RUN 的记忆目标仍由 actor API 逐次拒绝。
    此处不兑现访问凭据，也不代替 workspace 的操作授权。
    """
    return frozenset({MTPVerb.RUN.value, *(_OPERATION_VERBS[op] for op in operations)})
