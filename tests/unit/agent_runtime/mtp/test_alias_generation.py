"""
MTP 别名系统测试 (Section 2.3)

测试覆盖:
- MemoryAtom.get_alias() 的存储别名优先逻辑

别名构建与 Workspace 内唯一性由 ``AliasGenerator`` 负责，见
``tests/unit/engines/generation/test_alias.py``。

对应设计文档: MemoryToolProtocol.md Section 2.3
"""

from unittest.mock import MagicMock

from hivememory.core.models import MemoryAtom

# ========== Koakuma 别名偏好测试 ==========


class TestKoakumaAliasPreference:
    """测试 MemoryAtom.get_alias() 的存储别名优先逻辑"""

    def test_fallback_when_alias_none(self):
        """alias 为 None 时 fallback 到运行时生成"""
        mem = MagicMock()
        mem.index.alias = None
        mem.index.memory_type.value = "FACT"
        mem.index.title = "API Specification"

        result = MemoryAtom.get_alias(mem)
        assert result == "fact_api_specification"

    def test_fallback_when_alias_empty(self):
        """alias 为空字符串时 fallback 到运行时生成"""
        mem = MagicMock()
        mem.index.alias = ""
        mem.index.memory_type.value = "CODE_SNIPPET"
        mem.index.title = "Parse Date"

        result = MemoryAtom.get_alias(mem)
        assert result == "code_parse_date"
