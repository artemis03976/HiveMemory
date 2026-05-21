from unittest.mock import MagicMock

from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_uses_injected_dependencies():
    config = HiveMemoryConfig()
    local_bus = AliceBus()
    prompt_assembler = AgentPromptAssembler(config.koakuma)
    mtp_executor = MagicMock()

    runtime = AgentRuntime(
        local_bus=local_bus,
        prompt_assembler=prompt_assembler,
        mtp_executor=mtp_executor,
        config=config,
    )

    assert runtime._mtp_executor is mtp_executor
    assert runtime._loop_executor._local_bus is local_bus
    assert runtime._loop_executor._mtp_executor is mtp_executor
    assert runtime._frame_scheduler._prompt_assembler is prompt_assembler

