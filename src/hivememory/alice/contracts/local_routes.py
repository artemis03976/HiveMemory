"""Alice 子系统内部路由常量 — 仅供 AliceBus 本地挂载与调用。"""


class AliceLocalRoutes:
    RUN_AGENT = "agent.run"
    RUN_AGENT_STREAM = "agent.run_stream"
    REGISTER_PRERETRIEVAL_ALIASES = "runtime.register_preretrieval_aliases"
    GET_INTERACTION_STATE = "runtime.get_interaction_state"

    ALL = (
        RUN_AGENT,
        RUN_AGENT_STREAM,
        REGISTER_PRERETRIEVAL_ALIASES,
        GET_INTERACTION_STATE,
    )
