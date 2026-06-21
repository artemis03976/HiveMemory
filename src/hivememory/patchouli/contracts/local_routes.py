"""Patchouli 子系统内部路由常量 — 仅供 PatchouliBus 本地挂载与调用。"""


class PatchouliLocalRoutes:
    INGESTION_SUBMIT_INTERACTION = "ingestion.submit_interaction"
    ANALYZE_AND_RETRIEVE = "passive.analyze_and_retrieve"
    MEMORY_RETRIEVE = "memory.retrieve"
    MEMORY_RETRIEVE_BY_ALIASES = "memory.retrieve_by_aliases"
    MEMORY_RECORD_HIT = "memory.record_hit"
    MEMORY_RECORD_CITATION = "memory.record_citation"
    MEMORY_RECORD_FEEDBACK = "memory.record_feedback"
    MEMORY_REVIVE = "memory.revive"
    REFRESH_MEMORY_VITALITY = "lifecycle.refresh_memory_vitality"
    LIFECYCLE_RUN_GARDENING_ONCE = "lifecycle.run_gardening_once"
    GENERATION_SUBMIT_ARCHIVE = "generation.submit_archive"
    GENERATION_SUBMIT_ACTIVE = "generation.submit_active"
    GENERATION_EXECUTE_SPEC = "generation.execute_spec"
    MEMORY_TASK_SUBMIT_GENERATION = "memory_task.submit_generation"
    MEMORY_TASK_SUBMIT_GENERATION_MANY = "memory_task.submit_generation_many"
    MEMORY_GET = "memory.get"
    GET_AGENT_PROFILE = "memory.get_agent_profile"
    TOPIC_PREPARE = "topic.prepare"
    TOPIC_GET_SHORT_TERM = "topic.get_short_term"
    GET_ACTIVE_TOPICS_SNAPSHOTS = "librarian.get_active_topics_snapshots"
    TOPIC_MANUAL_ARCHIVE = "topic.manual_archive"
    TOPIC_EVICT = "topic.evict"
    TOPIC_DISCARD_IF_EMPTY = "topic.discard_if_empty"
    PREPARE_AGENT_RUN = "service.prepare_agent_run"
    FINALIZE_AGENT_RUN = "service.finalize_agent_run"
    CLEANUP_PREPARED_AGENT_RUN = "service.cleanup_prepared_agent_run"
    MANUAL_ARCHIVE_TOPIC = TOPIC_MANUAL_ARCHIVE
    PREPARE_TOPIC = TOPIC_PREPARE
    SUBMIT_INTERACTION = INGESTION_SUBMIT_INTERACTION

    ALL = (
        INGESTION_SUBMIT_INTERACTION,
        ANALYZE_AND_RETRIEVE,
        MEMORY_RETRIEVE,
        MEMORY_RETRIEVE_BY_ALIASES,
        MEMORY_RECORD_HIT,
        MEMORY_RECORD_CITATION,
        MEMORY_RECORD_FEEDBACK,
        MEMORY_REVIVE,
        REFRESH_MEMORY_VITALITY,
        LIFECYCLE_RUN_GARDENING_ONCE,
        GENERATION_SUBMIT_ARCHIVE,
        GENERATION_SUBMIT_ACTIVE,
        GENERATION_EXECUTE_SPEC,
        MEMORY_TASK_SUBMIT_GENERATION,
        MEMORY_TASK_SUBMIT_GENERATION_MANY,
        MEMORY_GET,
        GET_AGENT_PROFILE,
        TOPIC_PREPARE,
        TOPIC_GET_SHORT_TERM,
        GET_ACTIVE_TOPICS_SNAPSHOTS,
        TOPIC_MANUAL_ARCHIVE,
        TOPIC_EVICT,
        TOPIC_DISCARD_IF_EMPTY,
        PREPARE_AGENT_RUN,
        FINALIZE_AGENT_RUN,
        CLEANUP_PREPARED_AGENT_RUN,
    )
