from hivememory.engines.perception import create_perception_layer
from hivememory.engines.perception.relay_controller import (
    NoOpRelayController,
    create_relay_controller,
)
from hivememory.engines.perception.semantic_flow_perception_layer import NullPerceptionLayer
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
from hivememory.system.config import MemoryPerceptionConfig, RelayControllerConfig


def test_disabled_relay_config_returns_noop_controller():
    controller = create_relay_controller(RelayControllerConfig(enable=False))

    assert isinstance(controller, NoOpRelayController)
    assert controller.generate_summary([], previous_summary="keep") == "keep"


def test_disabled_perception_config_returns_null_layer():
    config = MemoryPerceptionConfig()
    config.engine.enable = False

    layer = create_perception_layer(
        config,
        short_term_store=ShortTermMemoryStore(),
        interaction_journal=InMemoryInteractionApplyJournal(),
    )

    assert isinstance(layer, NullPerceptionLayer)
