"""MTP SEARCH filter type 映射测试。"""

import pytest

from hivememory.core.models import MemoryType
from hivememory.core.mtp.parser import MTPFilterParser


class TestMTPFilterTypeMap:
    """MTP SEARCH 的 type 过滤别名到记忆类型的映射。"""

    @pytest.mark.parametrize("value", ["resource", "res"])
    def test_resource_filter(self, value):
        """type:resource 与简写 type:res 都过滤到资源记忆。"""
        filters, warnings = MTPFilterParser().parse(f"type:{value}")
        assert filters is not None
        assert filters.memory_type == MemoryType.RESOURCE
        assert warnings == []

    def test_historical_type_alias_only_warns(self):
        """只作历史取值的类型没有别名：按未知类型给出警告，不加类型过滤。"""
        filters, warnings = MTPFilterParser().parse("type:wip tag:test")
        assert filters is not None
        assert filters.memory_type is None
        assert [w.message_key for w in warnings] == ["mtp.filter.unknown_type"]

    def test_agent_filter_keeps_provenance_separate_from_actor_scope(self):
        """捕获 agent 业务过滤被误写成授权 ActorIdentity、绕开统一 scope 的回归。"""
        filters, warnings = MTPFilterParser().parse("agent:researcher")

        assert filters is not None
        assert filters.source_agent_id == "researcher"
        assert warnings == []
