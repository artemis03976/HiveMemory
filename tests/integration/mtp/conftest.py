"""
MTP 链路集成测试共享 Fixtures。

KoakumaRuntime 已重构为 bus-based 架构 (bus, config, alias_resolver)。
集成测试通过 MockAsyncBus 路由到 mock 服务，验证 Koakuma 与各组件
（parser、handler、compiler、alias resolver、syscall 执行器）的协作链路。
"""

import asyncio
from typing import Optional
from unittest.mock import MagicMock

import pytest

from hivememory.agent_runtime.aliases import KoakumaAtomCache, RuntimeAliasResolver
from hivememory.agent_runtime.models import MTPExecutionContext
from hivememory.agent_runtime.mtp.runtime import KoakumaRuntime
from hivememory.agent_runtime.pending_atom import PendingAtomRuntime
from hivememory.core.mtp import MTP_LEFT_DELIMITER, MTP_RIGHT_DELIMITER
from hivememory.core.protocol.models import MTPExecutionResult
from hivememory.prompts.mtp import MTPPromptBuilder
from hivememory.system.config import KoakumaConfig
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.async_bus import AsyncSystemBus
from tests.helpers.workspace import make_runtime_scope


class MockAsyncBus(AsyncSystemBus):
    """
    测试用异步总线替身。

    路由 bus.request(route, ...) 到内部 mock 服务:
      - "storage.get_memory" → _mock_storage.get_memory
      - "memory.retrieve_by_aliases" → _mock_storage.get_memory_by_alias
      - "retrieval.retrieve" / "memory.retrieve" → _mock_retrieval.retrieve

    测试中通过 bus._mock_storage / bus._mock_retrieval 配置 mock 行为。
    """

    def __init__(
        self,
        mock_storage: Optional[MagicMock] = None,
        mock_retrieval: Optional[MagicMock] = None,
        mock_generation: Optional[MagicMock] = None,
    ):
        super().__init__()
        self._mock_storage = mock_storage or MagicMock()
        self._mock_retrieval = mock_retrieval or MagicMock()
        self._mock_generation = mock_generation or MagicMock()
        self._memory_citations = []

        self.register("storage.get_memory", self._handle_get_memory)
        self.register("retrieval.retrieve", self._handle_retrieve)
        self.register("memory.retrieve", self._handle_retrieve)
        self.register("memory.retrieve_by_aliases", self._handle_retrieve_by_aliases)
        self.register(GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE, self._handle_retrieve)
        self.register(
            GlobalRoutes.PATCHOULI_MEMORY_RETRIEVE_BY_ALIASES,
            self._handle_retrieve_by_aliases,
        )
        self.register(
            GlobalRoutes.PATCHOULI_RECORD_MEMORY_CITATION,
            self._handle_record_memory_citation,
        )
        self.register("generation.process", self._handle_generation_process)
        self.register("perception.route_and_ingest", self._handle_route_and_ingest)

    async def _handle_get_memory(self, *args, **kwargs):
        return self._mock_storage.get_memory(*args, **kwargs)

    async def _handle_retrieve_by_aliases(self, *args, **kwargs):
        from hivememory.core.protocol.models import RetrievalResponse

        aliases = kwargs.get("aliases")
        if aliases is None and args:
            aliases = args[0]
        identity = kwargs.get("identity")
        user_id = getattr(identity, "user_id", None)

        memories = []
        for alias in aliases or []:
            atom = self._mock_storage.get_memory_by_alias(
                alias=alias,
                user_id=user_id,
            )
            if atom is not None:
                memories.append(atom)
        return RetrievalResponse(memories=memories, memories_count=len(memories))

    async def _handle_retrieve(self, *args, **kwargs):
        return self._mock_retrieval.retrieve(**kwargs)

    async def _handle_generation_process(self, *args, **kwargs):
        return self._mock_generation.process(*args, **kwargs)

    async def _handle_record_memory_citation(self, *args, **kwargs):
        self._memory_citations.append(
            {
                "memory_id": kwargs.get("memory_id"),
                "source": kwargs.get("source"),
            }
        )
        return {"success": True}

    async def _handle_route_and_ingest(self, *args, **kwargs):
        return None


def make_mock_bus(
    mock_storage: Optional[MagicMock] = None,
    mock_retrieval: Optional[MagicMock] = None,
    mock_generation: Optional[MagicMock] = None,
) -> MockAsyncBus:
    """工厂函数: 创建配置好的异步 mock bus。"""
    return MockAsyncBus(
        mock_storage=mock_storage,
        mock_retrieval=mock_retrieval,
        mock_generation=mock_generation,
    )


def make_runtime_alias_resolver(bus: MockAsyncBus) -> RuntimeAliasResolver:
    return RuntimeAliasResolver(
        pending_runtime=PendingAtomRuntime(),
        atom_cache=KoakumaAtomCache(),
        bus=bus,
    )


def make_koakuma_runtime(bus: MockAsyncBus, config=None) -> KoakumaRuntime:
    return KoakumaRuntime(
        bus=bus,
        config=config or KoakumaConfig(),
        alias_resolver=make_runtime_alias_resolver(bus),
    )


def normalize_worker_agent_mtp_output(text: str) -> str:
    """模拟 WorkerAgent 在检测到 MTP 指令后补全右定界符的输出。"""
    last_open = text.rfind(MTP_LEFT_DELIMITER)
    if last_open == -1:
        return text
    if MTP_RIGHT_DELIMITER in text[last_open:]:
        return text
    return text.rstrip() + " " + MTP_RIGHT_DELIMITER


def simulate_kernel_loop_single(koakuma: KoakumaRuntime, agent_text: str) -> MTPExecutionResult:
    result = asyncio.run(
        koakuma.intercept_and_execute(
            normalize_worker_agent_mtp_output(agent_text),
            context=MTPExecutionContext(runtime_scope=make_runtime_scope()),
        )
    )
    assert result is not None, f"Kernel Loop 未检测到 MTP 指令。Agent 文本: {agent_text!r}"
    return result


def build_resumed_history(agent_prefix: str, mtp_result: MTPExecutionResult) -> str:
    return agent_prefix + mtp_result.formatted_response


@pytest.fixture
def koakuma() -> KoakumaRuntime:
    bus = make_mock_bus()
    return make_koakuma_runtime(bus, KoakumaConfig())


@pytest.fixture
def mtp_prompt_en() -> str:
    return MTPPromptBuilder(language="en").build()


@pytest.fixture
def mtp_prompt_zh() -> str:
    return MTPPromptBuilder(language="zh").build()
