"""Canonical public route names shared by subsystem and global route views."""


class RouteNames:
    """Single source of truth for public GlobalSystemBus route strings."""

    PATCHOULI_PASSIVE_ANALYZE_AND_RETRIEVE = "patchouli.public.passive.analyze_and_retrieve"
    PATCHOULI_SUBMIT_INTERACTION = "patchouli.public.submit_interaction"
    PATCHOULI_MEMORY_RETRIEVE = "patchouli.public.memory.retrieve"
    PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES = "patchouli.public.memory.retrieve_by_aliases"
    PATCHOULI_GET_AGENT_PROFILE = "patchouli.public.get_agent_profile"
    PATCHOULI_PREPARE_AGENT_RUN = "patchouli.public.prepare_agent_run"
    PATCHOULI_FINALIZE_AGENT_RUN = "patchouli.public.finalize_agent_run"
    PATCHOULI_CLEANUP_PREPARED_AGENT_RUN = "patchouli.public.cleanup_prepared_agent_run"
    PATCHOULI_MANUAL_ARCHIVE_TOPIC = "patchouli.public.manual_archive_topic"
    PATCHOULI_RECORD_MEMORY_CITATION = "patchouli.public.record_memory_citation"

    ALICE_RUN_AGENT = "alice.public.run_agent"
    ALICE_RUN_AGENT_STREAM = "alice.public.run_agent_stream"
