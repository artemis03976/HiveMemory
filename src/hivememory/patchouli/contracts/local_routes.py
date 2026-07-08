"""Patchouli subsystem-local route constants."""


class PatchouliLocalRoutes:
    """Internal PatchouliBus route names.

    Local routes describe composable Patchouli primitives. Public workflows such
    as prepare/finalize agent run belong to Patchouli public routes instead.
    """

    INGESTION_SUBMIT_INTERACTION = "ingestion.submit_interaction"

    MEMORY_CREATE = "memory.create"
    MEMORY_LIST = "memory.list"
    MEMORY_GET = "memory.get"
    MEMORY_UPDATE = "memory.update"
    MEMORY_DELETE = "memory.delete"
    MEMORY_RETRIEVE = "memory.retrieve"
    MEMORY_RETRIEVE_BY_ALIASES = "memory.retrieve_by_aliases"
    MEMORY_RECORD_HIT = "memory.record_hit"
    MEMORY_RECORD_CITATION = "memory.record_citation"
    MEMORY_RECORD_FEEDBACK = "memory.record_feedback"
    MEMORY_REVIVE = "memory.revive"

    REFRESH_MEMORY_VITALITY = "lifecycle.refresh_memory_vitality"
    LIFECYCLE_RUN_GARDENING_ONCE = "lifecycle.run_gardening_once"

    GENERATION_SUBMIT_SETTLEMENT = "generation.submit_settlement"
    GENERATION_SUBMIT_ACTIVE = "generation.submit_active"
    GENERATION_EXECUTE_SPEC = "generation.execute_spec"

    MEMORY_TASK_SUBMIT_GENERATION = "memory_task.submit_generation"
    MEMORY_TASK_SUBMIT_GENERATION_MANY = "memory_task.submit_generation_many"
    MEMORY_TASK_LIST = "memory_task.list"
    MEMORY_TASK_GET = "memory_task.get"
    MEMORY_TASK_CANCEL = "memory_task.cancel"
    MEMORY_TASK_WAIT = "memory_task.wait"
    MEMORY_TASK_WAIT_MANY = "memory_task.wait_many"
    MEMORY_TASK_WAIT_ALL = "memory_task.wait_all"

    GET_AGENT_PROFILE = "memory.get_agent_profile"

    TOPIC_PREPARE = "topic.prepare"
    TOPIC_GET = "topic.get"
    TOPIC_LIST_ACTIVE = "topic.list_active"
    TOPIC_MANUAL_SETTLE = "topic.manual_settle"
    TOPIC_EVICT = "topic.evict"
    TOPIC_DISCARD_IF_EMPTY = "topic.discard_if_empty"

    RUNTIME_MODELS_WARMUP = "runtime.models.warmup"
    RUNTIME_MODELS_READY = "runtime.models.ready"
    RUNTIME_STORAGE_HEALTH = "runtime.storage_health"

    ALL = (
        INGESTION_SUBMIT_INTERACTION,
        MEMORY_CREATE,
        MEMORY_LIST,
        MEMORY_GET,
        MEMORY_UPDATE,
        MEMORY_DELETE,
        MEMORY_RETRIEVE,
        MEMORY_RETRIEVE_BY_ALIASES,
        MEMORY_RECORD_HIT,
        MEMORY_RECORD_CITATION,
        MEMORY_RECORD_FEEDBACK,
        MEMORY_REVIVE,
        REFRESH_MEMORY_VITALITY,
        LIFECYCLE_RUN_GARDENING_ONCE,
        GENERATION_SUBMIT_SETTLEMENT,
        GENERATION_SUBMIT_ACTIVE,
        GENERATION_EXECUTE_SPEC,
        MEMORY_TASK_SUBMIT_GENERATION,
        MEMORY_TASK_SUBMIT_GENERATION_MANY,
        MEMORY_TASK_LIST,
        MEMORY_TASK_GET,
        MEMORY_TASK_CANCEL,
        MEMORY_TASK_WAIT,
        MEMORY_TASK_WAIT_MANY,
        MEMORY_TASK_WAIT_ALL,
        GET_AGENT_PROFILE,
        TOPIC_PREPARE,
        TOPIC_GET,
        TOPIC_LIST_ACTIVE,
        TOPIC_MANUAL_SETTLE,
        TOPIC_EVICT,
        TOPIC_DISCARD_IF_EMPTY,
        RUNTIME_MODELS_WARMUP,
        RUNTIME_MODELS_READY,
        RUNTIME_STORAGE_HEALTH,
    )
