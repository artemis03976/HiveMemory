import pytest

from hivememory.config.patchouli import GenerationTranscriptConfig, RelayControllerConfig
from hivememory.engines.perception import create_fold_summary_writer
from hivememory.engines.perception.fold_summary import NoOpFoldSummaryWriter


@pytest.mark.asyncio
async def test_disabled_relay_config_returns_noop_controller():
    controller = create_fold_summary_writer(
        RelayControllerConfig(enable=False), transcript_config=GenerationTranscriptConfig()
    )

    assert isinstance(controller, NoOpFoldSummaryWriter)
    assert await controller.generate_summary([], previous_summary="keep") == "keep"
