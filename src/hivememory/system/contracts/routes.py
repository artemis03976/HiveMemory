"""全局路由常量 — 跨子系统的公共路由注册表。"""

from hivememory.alice.contracts.public_routes import AliceRoutes
from hivememory.patchouli.contracts.public_routes import PatchouliRoutes


class GlobalRoutes:
    """所有可通过 GlobalSystemBus 访问的公开路由。"""

    PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE = PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE
    PATCHOULI_SUBMIT_INTERACTION = PatchouliRoutes.SUBMIT_INTERACTION
    PATCHOULI_MEMORY_RETRIEVE = PatchouliRoutes.MEMORY_RETRIEVE
    PATCHOULI_MEMORY_GET_BY_ALIAS = PatchouliRoutes.MEMORY_GET_BY_ALIAS
    PATCHOULI_GET_AGENT_PROFILE = PatchouliRoutes.GET_AGENT_PROFILE
    PATCHOULI_PREPARE_AGENT_RUN = PatchouliRoutes.PREPARE_AGENT_RUN
    PATCHOULI_FINALIZE_AGENT_RUN = PatchouliRoutes.FINALIZE_AGENT_RUN
    PATCHOULI_CLEANUP_PREPARED_AGENT_RUN = PatchouliRoutes.CLEANUP_PREPARED_AGENT_RUN
    ALICE_RUN_AGENT = AliceRoutes.RUN_AGENT
    ALICE_RUN_AGENT_STREAM = AliceRoutes.RUN_AGENT_STREAM
    ALICE_REGISTER_PRERETRIEVAL_ALIASES = AliceRoutes.REGISTER_PRERETRIEVAL_ALIASES
