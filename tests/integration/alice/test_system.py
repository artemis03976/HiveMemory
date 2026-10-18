"""AliceSystem 启停控制真实 CPU 端口准入；LLM 边界使用确定性替身。"""

import pytest

from hivememory.agent_runtime import GenerationResult, LLMClient
from hivememory.alice import AliceSystem
from hivememory.alice.interface.cpu import AliceUnavailableError
from hivememory.config.alice import AliceConfig
from tests.helpers.chat_handoff import make_input_manifest
from tests.helpers.operations import OperationsHarness
from tests.helpers.workspace import make_access_composition, make_actor_access_record


@pytest.mark.asyncio
async def test_system_lifecycle_controls_cpu_execution(monkeypatch):
    async def generate(_self, messages, **kwargs):
        return GenerationResult(text="完成")

    monkeypatch.setattr(LLMClient, "generate_async", generate)
    operations = OperationsHarness()
    system = AliceSystem(config=AliceConfig(), operation_entry=operations.entry)
    access = await make_access_composition([make_actor_access_record()]).authenticate_thread()

    def execute():
        return system.cpu_port.execute(
            make_input_manifest(), access=access, generation_options=None, stream=False
        )

    await system.start()
    result = [item async for item in execute().output]
    assert [(item.status, item.final_text) for item in result] == [("completed", "完成")]
    await system.stop()
    with pytest.raises(AliceUnavailableError, match="not running"):
        await anext(execute().output)
