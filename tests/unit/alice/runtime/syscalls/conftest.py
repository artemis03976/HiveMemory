"""syscalls 测试共享 fixture 与 helper。"""

import asyncio
from typing import Optional
from unittest.mock import MagicMock

import pytest

from hivememory.system.config import KoakumaConfig
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.core.protocol.models import MTPExecutionResult
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class MockAsyncBus(AsyncSystemBus):
    def __init__(self, mock_storage: Optional[MagicMock] = None, mock_retrieval: Optional[MagicMock] = None, mock_generation: Optional[MagicMock] = None):
        super().__init__()
        self._mock_storage = mock_storage or MagicMock()
        self._mock_retrieval = mock_retrieval or MagicMock()
        self._mock_generation = mock_generation or MagicMock()

        self.register("storage.get_memory", self._handle_get_memory)
        self.register("storage.get_memory_by_alias", self._handle_get_memory_by_alias)
        self.register("memory.get_memory_by_alias", self._handle_get_memory_by_alias)
        self.register("retrieval.retrieve", self._handle_retrieve)
        self.register("memory.retrieve", self._handle_retrieve)
        self.register("generation.process", self._handle_generation_process)
        self.register("perception.route_and_ingest", self._handle_route_and_ingest)

    async def _handle_get_memory(self, *args, **kwargs):
        return self._mock_storage.get_memory(*args, **kwargs)

    async def _handle_get_memory_by_alias(self, *args, **kwargs):
        return self._mock_storage.get_memory_by_alias(**kwargs)

    async def _handle_retrieve(self, *args, **kwargs):
        return self._mock_retrieval.retrieve(**kwargs)

    async def _handle_generation_process(self, *args, **kwargs):
        return self._mock_generation.process(*args, **kwargs)

    async def _handle_route_and_ingest(self, *args, **kwargs):
        return None


def make_mock_bus(mock_storage: Optional[MagicMock] = None, mock_retrieval: Optional[MagicMock] = None, mock_generation: Optional[MagicMock] = None) -> MockAsyncBus:
    return MockAsyncBus(mock_storage=mock_storage, mock_retrieval=mock_retrieval, mock_generation=mock_generation)


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    return KoakumaRuntime(bus=make_mock_bus(), config=KoakumaConfig())


@pytest.fixture
def mtp_prompt_en() -> str:
    return MTPPromptBuilder(language="en").build()


@pytest.fixture
def mtp_prompt_zh() -> str:
    return MTPPromptBuilder(language="zh").build()


def simulate_kernel_loop_single(koakuma: KoakumaRuntime, agent_text: str) -> MTPExecutionResult:
    result = asyncio.run(koakuma.intercept_and_execute(agent_text))
    assert result is not None, f"Kernel Loop 未检测到 MTP 指令。Agent 文本: {agent_text!r}"
    return result


def build_resumed_history(agent_prefix: str, mtp_result: MTPExecutionResult) -> str:
    return agent_prefix + mtp_result.formatted_response
