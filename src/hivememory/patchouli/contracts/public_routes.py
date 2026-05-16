"""Patchouli 子系统公开路由常量 — 可通过桥接器暴露到 GlobalSystemBus 的能力。"""


class PatchouliRoutes:
    PASSIVE_ANALYZE_AND_RETRIEVE = "patchouli.public.passive.analyze_and_retrieve"
    PASSIVE_HANDLE_HOT = PASSIVE_ANALYZE_AND_RETRIEVE
    SUBMIT_INTERACTION = "patchouli.public.submit_interaction"
    MEMORY_RETRIEVE = "patchouli.public.memory.retrieve"
    MEMORY_GET_BY_ALIAS = "patchouli.public.memory.get_memory_by_alias"
