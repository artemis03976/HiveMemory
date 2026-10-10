"""逐次授权的 workspace actor API，不导入 root 管理能力。"""

from hivememory.workspace.capability.actor.agent_profiles import ActorAgentService
from hivememory.workspace.capability.actor.memory import ActorMemoryService

__all__ = ["ActorAgentService", "ActorMemoryService"]
