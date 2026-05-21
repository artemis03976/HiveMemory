"""Alice public routes exposed through GlobalSystemBus."""

from hivememory.system.contracts.route_names import RouteNames


class AliceRoutes:
    RUN_AGENT = RouteNames.ALICE_RUN_AGENT
    RUN_AGENT_STREAM = RouteNames.ALICE_RUN_AGENT_STREAM
