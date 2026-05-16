"""syscalls 测试共享 fixture 与 helper。"""

from typing import Optional
from unittest.mock import MagicMock

import pytest

from hivememory.system.config import KoakumaConfig
from hivememory.alice.runtime.koakuma import KoakumaRuntime
from hivememory.patchouli.protocol.models import MTPExecutionResult
from hivememory.prompts.mtp import MTPPromptBuilder


class MockSystemBus:
    def __init__(self, mock_storage: Optional[MagicMock] = None, mock_retrieval: Optional[MagicMock] = None, mock_generation: Optional[MagicMock] = None):
        self._mock_storage = mock_storage or MagicMock()
        self._mock_retrieval = mock_retrieval or MagicMock()
        self._mock_generation = mock_generation or MagicMock()

    def request(self, route: str, *args, **kwargs):
        if route == "storage.get_memory":
            return self._mock_storage.get_memory(*args, **kwargs)
        if route in ("storage.get_memory_by_alias", "memory.get_memory_by_alias"):
            return self._mock_storage.get_memory_by_alias(**kwargs)
        if route in ("retrieval.retrieve", "memory.retrieve"):
            return self._mock_retrieval.retrieve(**kwargs)
        if route == "generation.process":
            return self._mock_generation.process(*args, **kwargs)
        if route == "perception.route_and_ingest":
            return None
        raise ValueError(f"MockSystemBus: unknown route '{route}'")

    def subscribe(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


def make_mock_bus(mock_storage: Optional[MagicMock] = None, mock_retrieval: Optional[MagicMock] = None, mock_generation: Optional[MagicMock] = None) -> MockSystemBus:
    return MockSystemBus(mock_storage=mock_storage, mock_retrieval=mock_retrieval, mock_generation=mock_generation)


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
    result = koakuma.intercept_and_execute(agent_text)
    assert result is not None, f"Kernel Loop 未检测到 MTP 指令。Agent 文本: {agent_text!r}"
    return result


def build_resumed_history(agent_prefix: str, mtp_result: MTPExecutionResult) -> str:
    return agent_prefix + mtp_result.formatted_response
