"""
MTP 链路测试共享 Fixtures

提供 mock Async bus 工厂，用于替代旧的直接服务注入模式。
KoakumaRuntime 已重构为 bus-based 架构 (bus, config)，
测试需要通过 mock bus 路由到 mock 服务。

作者: HiveMemory Team
"""

from unittest.mock import MagicMock
from typing import Optional

import pytest

from hivememory.system.runtime.bus.async_bus import AsyncSystemBus


class MockAsyncBus(AsyncSystemBus):
    """
    测试用异步总线替身

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

        self.register("storage.get_memory", self._handle_get_memory)
        self.register("retrieval.retrieve", self._handle_retrieve)
        self.register("memory.retrieve", self._handle_retrieve)
        self.register("memory.retrieve_by_aliases", self._handle_retrieve_by_aliases)
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

    async def _handle_route_and_ingest(self, *args, **kwargs):
        return None


def make_mock_bus(
    mock_storage: Optional[MagicMock] = None,
    mock_retrieval: Optional[MagicMock] = None,
    mock_generation: Optional[MagicMock] = None,
) -> MockAsyncBus:
    """工厂函数: 创建配置好的异步 mock bus"""
    return MockAsyncBus(
        mock_storage=mock_storage,
        mock_retrieval=mock_retrieval,
        mock_generation=mock_generation,
    )
