"""workspace 能力层：in-process 的 workspace server API（A2 §1.2 / 宪章 §5.3）。

由 ``system/application`` 的资源能力部分改造而来：actor 经 HTTP/MTP/外部
adapter 归一化后调用能力方法；能力层在 backing 调用前执行 operation 授权，
并作为 client 调用 Patchouli backing 路由（第二层 client-server）。
chat 任务进程的编排位于 ``workspace.process``，被动摄入与就绪检查属于
System 级能力，均不在本子包。
"""

from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.assets import WorkspaceAssetApplicationService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.memory import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
from hivememory.workspace.capability.memory_tasks import MemoryTaskApplicationService
from hivememory.workspace.capability.topic import TopicApplicationService

__all__ = [
    "AgentApplicationService",
    "BusCanonicalReadBackend",
    "MemoryApplicationService",
    "MemoryLifecycleUnavailableError",
    "MemoryNotFoundError",
    "MemoryTaskApplicationService",
    "TopicApplicationService",
    "WorkspaceAssetApplicationService",
]
