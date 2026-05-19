"""Patchouli 子系统内部路由常量 — 仅供 PatchouliBus 本地挂载与调用。"""


class PatchouliLocalRoutes:
    SUBMIT_INTERACTION = "librarian.submit_interaction"
    ANALYZE_AND_RETRIEVE = "passive.analyze_and_retrieve"
    MEMORY_RETRIEVE = "memory.retrieve"
    MEMORY_GET_BY_ALIAS = "memory.get_memory_by_alias"
    GET_AGENT_PROFILE = "memory.get_agent_profile"
    PREPARE_TOPIC = "librarian.prepare_topic"
    GET_ACTIVE_TOPICS_SNAPSHOTS = "librarian.get_active_topics_snapshots"
    PREPARE_AGENT_RUN = "service.prepare_agent_run"
    FINALIZE_AGENT_RUN = "service.finalize_agent_run"
    CLEANUP_PREPARED_AGENT_RUN = "service.cleanup_prepared_agent_run"
    MANUAL_ARCHIVE_TOPIC = "librarian.manual_archive_topic"

    ALL = (
        SUBMIT_INTERACTION,
        ANALYZE_AND_RETRIEVE,
        MEMORY_RETRIEVE,
        MEMORY_GET_BY_ALIAS,
        GET_AGENT_PROFILE,
        PREPARE_TOPIC,
        GET_ACTIVE_TOPICS_SNAPSHOTS,
        PREPARE_AGENT_RUN,
        FINALIZE_AGENT_RUN,
        CLEANUP_PREPARED_AGENT_RUN,
        MANUAL_ARCHIVE_TOPIC,
    )
