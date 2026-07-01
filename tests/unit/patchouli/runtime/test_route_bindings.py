from unittest.mock import Mock

from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.route_bindings import build_patchouli_route_bindings


def test_build_patchouli_route_bindings_covers_all_declared_local_routes():
    runtime = Mock()
    service = Mock()

    route_names = [
        route
        for route, _handler in build_patchouli_route_bindings(runtime, service)
    ]

    assert set(route_names) == set(PatchouliLocalRoutes.ALL)
    assert len(route_names) == len(PatchouliLocalRoutes.ALL)
