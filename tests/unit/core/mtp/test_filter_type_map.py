"""MTP SEARCH filter type 映射测试。"""

from hivememory.core.models import MemoryType
from hivememory.core.mtp.parser import MTPFilterParser


class TestMTPFilterTypeMap:
    """MTP Filter Type Map 测试 (AGENT_PROFILE 支持)"""

    def test_agent_profile_filter(self):
        """type:agent_profile 过滤器"""
        filters, warnings = MTPFilterParser().parse("type:agent_profile")
        assert filters is not None
        assert filters.memory_type == MemoryType.AGENT_PROFILE
        assert warnings == []

    def test_agent_alias_filter(self):
        """type:agent 过滤器 (别名)"""
        filters, warnings = MTPFilterParser().parse("type:agent")
        assert filters is not None
        assert filters.memory_type == MemoryType.AGENT_PROFILE
        assert warnings == []

    def test_agent_filter_keeps_provenance_separate_from_actor_scope(self):
        """捕获 agent 业务过滤被误写成授权 ActorIdentity、绕开统一 scope 的回归。"""
        filters, warnings = MTPFilterParser().parse("agent:researcher")

        assert filters is not None
        assert filters.source_agent_id == "researcher"
        assert warnings == []
