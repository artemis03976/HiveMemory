"""
MTP 链路测试共享 Fixtures

提供 mock SystemBus 工厂，用于替代旧的直接服务注入模式。
KoakumaRuntime 已重构为 bus-based 架构 (bus, config)，
测试需要通过 mock bus 路由到 mock 服务。

作者: HiveMemory Team
"""

from unittest.mock import MagicMock
from typing import Optional

import pytest


class MockSystemBus:
    """
    测试用 SystemBus 替身

    路由 bus.request(route, ...) 到内部 mock 服务:
      - "storage.get_memory" → _mock_storage.get_memory
      - "storage.get_memory_by_alias" / "memory.get_memory_by_alias"
        → _mock_storage.get_memory_by_alias
      - "retrieval.retrieve" / "memory.retrieve" → _mock_retrieval.retrieve

    测试中通过 bus._mock_storage / bus._mock_retrieval 配置 mock 行为。
    """

    def __init__(
        self,
        mock_storage: Optional[MagicMock] = None,
        mock_retrieval: Optional[MagicMock] = None,
        mock_generation: Optional[MagicMock] = None,
    ):
        self._mock_storage = mock_storage or MagicMock()
        self._mock_retrieval = mock_retrieval or MagicMock()
        self._mock_generation = mock_generation or MagicMock()

    async def request(self, route: str, *args, **kwargs):
        if route == "storage.get_memory":
            return self._mock_storage.get_memory(*args, **kwargs)
        elif route in ("storage.get_memory_by_alias", "memory.get_memory_by_alias"):
            return self._mock_storage.get_memory_by_alias(**kwargs)
        elif route in ("retrieval.retrieve", "memory.retrieve"):
            return self._mock_retrieval.retrieve(**kwargs)
        elif route == "generation.process":
            return self._mock_generation.process(*args, **kwargs)
        elif route == "perception.route_and_ingest":
            return None
        else:
            raise ValueError(f"MockSystemBus: unknown route '{route}'")

    def subscribe(self, *args, **kwargs):
        pass

    def emit(self, *args, **kwargs):
        pass


def make_mock_bus(
    mock_storage: Optional[MagicMock] = None,
    mock_retrieval: Optional[MagicMock] = None,
    mock_generation: Optional[MagicMock] = None,
) -> MockSystemBus:
    """工厂函数: 创建配置好的 MockSystemBus"""
    return MockSystemBus(
        mock_storage=mock_storage,
        mock_retrieval=mock_retrieval,
        mock_generation=mock_generation,
    )
