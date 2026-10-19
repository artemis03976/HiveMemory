"""AliceSystem 启停控制真实 CPU 端口准入；LLM 边界使用确定性替身。"""

import pytest

from hivememory.alice import AliceSystem
from hivememory.alice.interface.cpu import AliceUnavailableError
from hivememory.config.alice import AliceConfig
from tests.helpers.alice import CollectedEvents
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.llm import ScriptedLLMService
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.mark.asyncio
async def test_system_lifecycle_controls_cpu_execution():
    operations = OperationsHarness()
    system = AliceSystem(
        config=AliceConfig(),
        operation_entry=operations.entry,
        llm_service=ScriptedLLMService("完成"),
    )
    access = await make_access_composition([make_actor_access_record()]).authenticate_thread()

    async def execute():
        return await system.cpu_port.execute(
            make_input_manifest(),
            access=access,
            generation_options={"model": "test-model"},
            events=CollectedEvents(),
        )

    await system.start()
    terminal = await execute()
    assert (terminal.status, terminal.final_text) == ("completed", "完成")
    await system.stop()
    with pytest.raises(AliceUnavailableError, match="not running"):
        await execute()
