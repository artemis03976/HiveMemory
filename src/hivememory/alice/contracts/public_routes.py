"""通过 GlobalSystemBus 暴露的 Alice 公开路由。"""

from hivememory.system.contracts.route_names import RouteNames


class AliceRoutes:
    RUN_AGENT = RouteNames.ALICE_RUN_AGENT
    RUN_AGENT_STREAM = RouteNames.ALICE_RUN_AGENT_STREAM
