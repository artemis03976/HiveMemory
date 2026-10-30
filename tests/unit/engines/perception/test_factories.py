import pytest

from hivememory.config.patchouli import GenerationTranscriptConfig, RelayControllerConfig
from hivememory.engines.perception import create_relay_controller
from hivememory.engines.perception.relay_controller import NoOpRelayController


@pytest.mark.asyncio
async def test_disabled_relay_config_returns_noop_controller():
    controller = create_relay_controller(
        RelayControllerConfig(enable=False), transcript_config=GenerationTranscriptConfig()
    )

    assert isinstance(controller, NoOpRelayController)
    assert await controller.generate_summary([], previous_summary="keep") == "keep"
