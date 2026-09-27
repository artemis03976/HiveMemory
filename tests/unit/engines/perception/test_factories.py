from hivememory.config.patchouli import RelayControllerConfig
from hivememory.engines.perception import create_relay_controller
from hivememory.engines.perception.relay_controller import NoOpRelayController


def test_disabled_relay_config_returns_noop_controller():
    controller = create_relay_controller(RelayControllerConfig(enable=False))

    assert isinstance(controller, NoOpRelayController)
    assert controller.generate_summary([], previous_summary="keep") == "keep"
