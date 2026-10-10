"""workspace 能力层：actor API 与共享 backing；root API 从 root 子包取得。"""

from hivememory.workspace.capability.agent_profiles import AgentApplicationService
from hivememory.workspace.capability.backing import BusCanonicalReadBackend
from hivememory.workspace.capability.memory import MemoryApplicationService

__all__ = ["AgentApplicationService", "BusCanonicalReadBackend", "MemoryApplicationService"]
