"""Patchouli 子系统域事件常量 — 可通过桥接器上抛到 GlobalSystemBus 的领域事件。"""


class PatchouliEvents:
    MEMORY_GENERATED = "patchouli.domain.memory_generated"
    TOPIC_EVICTED = "patchouli.domain.topic_evicted"
    OBSERVER_SESSION_FLUSHED = "patchouli.domain.observer_session_flushed"
