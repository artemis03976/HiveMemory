"""全局路由常量 — 跨子系统的公共路由注册表。"""

from hivememory.patchouli.contracts.public_routes import PatchouliRoutes


class GlobalRoutes:
    """所有可通过 GlobalSystemBus 访问的公开路由。"""

    PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE = PatchouliRoutes.PASSIVE_ANALYZE_AND_RETRIEVE
    PATCHOULI_PASSIVE_HANDLE_HOT = PatchouliRoutes.PASSIVE_HANDLE_HOT
    PATCHOULI_SUBMIT_INTERACTION = PatchouliRoutes.SUBMIT_INTERACTION
