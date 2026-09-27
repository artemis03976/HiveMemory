"""System 应用服务入口。

``passive_ingress`` / ``readiness`` 属于 System 级能力；chat 编排已迁至
``hivememory.alice.application``；资源能力部分已迁至
``hivememory.workspace.capability``（A2 §1.2），此处经迁移期 shim 继续导出，
A6 完成消费者切换后删除。
"""

from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.memory_service import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
from hivememory.system.application.memory_task_service import MemoryTaskApplicationService
from hivememory.system.application.passive_ingress_service import PassiveIngressService
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService

__all__ = [
    "AgentApplicationService",
    "MemoryApplicationService",
    "MemoryLifecycleUnavailableError",
    "MemoryNotFoundError",
    "MemoryTaskApplicationService",
    "PassiveIngressService",
    "SystemReadinessService",
    "TopicApplicationService",
]
