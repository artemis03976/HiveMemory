from hivememory.engines.perception import create_relay_controller
from hivememory.engines.perception.relay_controller import NoOpRelayController
from hivememory.system.config import RelayControllerConfig


def test_disabled_relay_config_returns_noop_controller():
    controller = create_relay_controller(RelayControllerConfig(enable=False))

    assert isinstance(controller, NoOpRelayController)
    assert controller.generate_summary([], previous_summary="keep") == "keep"
