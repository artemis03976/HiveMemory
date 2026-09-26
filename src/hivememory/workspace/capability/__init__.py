"""workspace 能力层：in-process 的 workspace server API（A2 §1.2 / 宪章 §5.3）。

由 ``system/application`` 的资源能力部分改造而来：actor 经 HTTP/MTP/外部
adapter 归一化后调用能力方法；能力层在 backing 调用前执行 operation 授权，
并作为 client 调用 Patchouli backing 路由（第二层 client-server）。
``chat_service`` / ``passive_ingress_service`` / ``readiness_service`` 仍留在
System；原 ``system/application`` 路径保留迁移期 re-export shim，A6 删除。

TODO(A5/A6)：本子包按过渡期分层导入白名单额外依赖 ``system.*`` 与
``patchouli.contracts``（A2 §8 D-2）；workspace 其余子包仍只依赖 core。
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
