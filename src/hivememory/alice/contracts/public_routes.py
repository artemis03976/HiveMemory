"""Alice 子系统公开路由常量 — 可通过桥接器暴露到 GlobalSystemBus 的能力。"""


class AliceRoutes:
    RUN_AGENT = "alice.run_agent"
    RUN_AGENT_STREAM = "alice.run_agent_stream"
    REGISTER_PRERETRIEVAL_ALIASES = "alice.runtime.register_preretrieval_aliases"
    GET_INTERACTION_STATE = "alice.runtime.get_interaction_state"
