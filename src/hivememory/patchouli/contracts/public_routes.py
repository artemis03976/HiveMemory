"""Patchouli 子系统公开路由常量 — 可通过桥接器暴露到 GlobalSystemBus 的能力。"""


class PatchouliRoutes:
    PASSIVE_ANALYZE_AND_RETRIEVE = "patchouli.public.passive.analyze_and_retrieve"
    SUBMIT_INTERACTION = "patchouli.public.submit_interaction"
    MEMORY_RETRIEVE = "patchouli.public.memory.retrieve"
    MEMORY_GET_BY_ALIAS = "patchouli.public.memory.get_memory_by_alias"
    PREPARE_AGENT_RUN = "patchouli.public.prepare_agent_run"
    FINALIZE_AGENT_RUN = "patchouli.public.finalize_agent_run"
    CLEANUP_PREPARED_AGENT_RUN = "patchouli.public.cleanup_prepared_agent_run"
    MANUAL_ARCHIVE_TOPIC = "patchouli.public.manual_archive_topic"
