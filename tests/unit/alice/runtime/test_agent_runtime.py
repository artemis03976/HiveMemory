from unittest.mock import MagicMock

from hivememory.alice.runtime.agent.runtime import AgentRuntime
from hivememory.alice.runtime.bus import AliceBus
from hivememory.agent_runtime.cache import KoakumaAtomCache
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.agent_runtime.resolver import RuntimeAliasResolver
from hivememory.prompts.assembler import AgentPromptAssembler
from hivememory.system.config import HiveMemoryConfig


def test_agent_runtime_uses_injected_dependencies():
    config = HiveMemoryConfig()
    local_bus = AliceBus()
    prompt_assembler = AgentPromptAssembler(config.koakuma)
    mtp_executor = MagicMock()
    alias_resolver = RuntimeAliasResolver(
        pending_runtime=PendingAtomRuntime(),
        atom_cache=KoakumaAtomCache(),
        bus=local_bus,
    )

    runtime = AgentRuntime(
        local_bus=local_bus,
        prompt_assembler=prompt_assembler,
        mtp_executor=mtp_executor,
        config=config,
        alias_resolver=alias_resolver,
    )

    orchestrator = runtime._orchestrator
    assert orchestrator._agent_profile_resolver._local_bus is local_bus
    assert orchestrator._loop_executor._mtp_executor is mtp_executor
    assert orchestrator._alias_resolver is alias_resolver
    assert orchestrator._frame_scheduler._prompt_assembler is prompt_assembler

