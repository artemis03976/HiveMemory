"""Alice 子系统公开路由常量 — 可通过桥接器暴露到 GlobalSystemBus 的能力。"""


class AliceRoutes:
    RUN_AGENT = "alice.public.run_agent"
    RUN_AGENT_STREAM = "alice.public.run_agent_stream"
    REGISTER_PRERETRIEVAL_ALIASES = "alice.public.register_preretrieval_aliases"
