"""
Renderer 单元测试

测试覆盖:
- FullContextRenderer: 统一模板渲染、截断逻辑
- CascadeContextRenderer: 瀑布式分级渲染 (Top-N 完整 + 其余 Index)
- CompactContextRenderer: 仅 Index 层渲染
- create_renderer 工厂函数
"""

import pytest
from datetime import datetime, timedelta

from hivememory.core.models import MemoryAtom, MemoryType, PayloadLayer, IndexLayer, MetaData, VerificationStatus
from hivememory.engines.retrieval.renderer import (
    FullContextRenderer,
    CascadeContextRenderer,
    CompactContextRenderer,
    create_renderer,
)
from hivememory.engines.retrieval.models import RenderFormat
from hivememory.system.config import (
    FullRendererConfig,
    CascadeRendererConfig,
    CompactRendererConfig,
)


class TestFullContextRenderer:
    """测试完整上下文渲染器"""

    def setup_method(self):
        self.renderer = FullContextRenderer(FullRendererConfig())

        # 创建测试记忆
        self.memory1 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 1",
                summary="This is the summary for test memory 1.",
                memory_type=MemoryType.FACT,
                tags=["test", "unit"]
            ),
            payload=PayloadLayer(content="This is the content of memory 1."),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(hours=2),
                confidence_score=0.95,
                verification_status=VerificationStatus.VERIFIED
            )
        )

        self.memory2 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 2",
                summary="This is the summary for test memory 2.",
                memory_type=MemoryType.CODE_SNIPPET,
                tags=["python"]
            ),
            payload=PayloadLayer(content="def hello():\n    print('world')"),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=5),
                confidence_score=0.8
            )
        )

    def test_empty_results(self):
        """测试空结果返回精简闭环提示"""
        from hivememory.engines.retrieval.renderer import _EMPTY_CONTEXT_NOTICE
        assert self.renderer.render([]) == _EMPTY_CONTEXT_NOTICE

    def test_content_truncation(self):
        """测试内容截断"""
        long_content = "Word " * 200
        memory = MemoryAtom(
            index=IndexLayer(title="Long", summary="This is a sufficiently long summary.", memory_type=MemoryType.FACT),
            payload=PayloadLayer(content=long_content),
            meta=MetaData(source_agent_id="test", user_id="u1")
        )

        config = FullRendererConfig(max_content_length=50)
        renderer = FullContextRenderer(config)
        output = renderer.render([memory])

        assert "部分内容已截断" in output
        assert len(output) < len(long_content) + 200

    def test_time_formatting(self):
        """测试时间格式化"""
        # 2小时前
        output = self.renderer.render([self.memory1])
        assert "小时前" in output

        # 5天前
        output = self.renderer.render([self.memory2])
        assert "天前" in output

        # 测试更早的时间（40天 = 1个月）
        old_memory = MemoryAtom(
            index=IndexLayer(
                title="Old Memory",
                summary="Old memory summary.",
                memory_type=MemoryType.FACT,
                tags=["old"]
            ),
            payload=PayloadLayer(content="Old content"),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=40)
            )
        )
        output = self.renderer.render([old_memory])
        assert "个月前" in output

    def test_confidence_formatting(self):
        """测试置信度格式化"""
        from hivememory.utils.memory_atom_renderer import MemoryAtomRenderer

        # 高置信度
        assert "(高)" in MemoryAtomRenderer._format_confidence(self.memory1)
        # 中置信度
        assert "(中)" in MemoryAtomRenderer._format_confidence(self.memory2)

    def test_with_config(self):
        """测试使用配置初始化"""
        config = FullRendererConfig(
            max_tokens=1000,
            max_content_length=100,
            stale_days=30
        )
        renderer = FullContextRenderer(config)

        assert renderer.max_tokens == 1000
        assert renderer.max_content_length == 100

    def test_unified_template_structure(self):
        """测试统一模板结构 (XML 结构 + MD 内容)"""
        output = self.renderer.render([self.memory1])

        # 统一 Header/Footer
        assert "<memory_context>" in output
        assert "</memory_context>" in output
        assert "帕秋莉" in output

        # Full item 模板
        assert "<memory alias=" in output
        assert "</memory>" in output
        assert "### Test Memory 1" in output
        assert "**类型**:" in output
        assert "**存档于**:" in output
        assert "**置信度**:" in output
        assert "**标签**:" in output
        assert "[完整内容]:" in output

        # Footer 中的 READ 指令
        assert "⟪ READ" in output


class TestCascadeContextRenderer:
    """测试瀑布式上下文渲染器"""

    def setup_method(self):
        """创建测试记忆"""
        self.memory1 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 1",
                summary="This is the summary for test memory 1.",
                memory_type=MemoryType.FACT,
                tags=["test", "unit"]
            ),
            payload=PayloadLayer(content="This is the content of memory 1. " * 10),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(hours=2),
                confidence_score=0.95
            )
        )

        self.memory2 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 2",
                summary="This is the summary for test memory 2.",
                memory_type=MemoryType.CODE_SNIPPET,
                tags=["python"]
            ),
            payload=PayloadLayer(content="def hello():\n    print('world')\n" * 5),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=5),
                confidence_score=0.8
            )
        )

        self.memory3 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 3",
                summary="This is the summary for test memory 3.",
                memory_type=MemoryType.REFLECTION,
                tags=["reflection"]
            ),
            payload=PayloadLayer(content="Reflection content here. " * 10),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=10),
                confidence_score=0.7
            )
        )

    def test_tiered_rendering(self):
        """测试 Top-1 完整渲染，其余降级"""
        config = CascadeRendererConfig(
            max_memory_tokens=2000,
            full_payload_count=1,
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 第一条应该是完整渲染 (<memory alias=...>)
        assert '<memory alias="fact_test_memory_1"' in output
        # 后续应该是 Index 视图 (<memory_index alias=...>)
        assert '<memory_index alias="code_test_memory_2"' in output or '<memory_index alias="reflection_test_memory_3"' in output

    def test_budget_truncation(self):
        """测试超出预算时降级为 Index 渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=600,
            full_payload_count=1,
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        assert len(output) > 0
        # 由于预算限制，应该有 Index 视图
        assert "memory_index" in output

    def test_budget_exhausted(self):
        """测试预算耗尽时返回精简闭环提示"""
        from hivememory.engines.retrieval.renderer import _EMPTY_CONTEXT_NOTICE
        config = CascadeRendererConfig(
            max_memory_tokens=200,
            full_payload_count=0,
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        assert output == _EMPTY_CONTEXT_NOTICE

    def test_empty_results(self):
        """测试空结果返回精简闭环提示"""
        from hivememory.engines.retrieval.renderer import _EMPTY_CONTEXT_NOTICE
        config = CascadeRendererConfig()
        renderer = CascadeContextRenderer(config)
        assert renderer.render([]) == _EMPTY_CONTEXT_NOTICE

    def test_all_full_payload(self):
        """测试所有记忆都完整渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=5000,
            full_payload_count=3,
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 所有记忆都应该是完整渲染
        assert "[完整内容]:" in output
        assert output.count("<memory alias=") == 3

    def test_full_payload_count_multiple(self):
        """测试 Top-2 完整渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=3000,
            full_payload_count=2,
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 前两条完整渲染
        assert '<memory alias="fact_test_memory_1"' in output
        assert '<memory alias="code_test_memory_2"' in output
        # 第三条降级为 Index
        assert '<memory_index alias="reflection_test_memory_3"' in output

    def test_unified_header_footer(self):
        """测试统一 Header/Footer"""
        config = CascadeRendererConfig(max_memory_tokens=5000)
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1])

        assert "<memory_context>" in output
        assert "</memory_context>" in output
        assert "帕秋莉" in output
        assert "⟪ READ" in output


class TestCompactContextRenderer:
    """测试紧凑上下文渲染器"""

    def setup_method(self):
        self.memory1 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 1",
                summary="This is the summary for test memory 1.",
                memory_type=MemoryType.FACT,
                tags=["test", "unit"]
            ),
            payload=PayloadLayer(content="Full content here"),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(hours=1),
                confidence_score=0.9
            )
        )

        self.memory2 = MemoryAtom(
            index=IndexLayer(
                title="Test Memory 2",
                summary="This is the summary for test memory 2.",
                memory_type=MemoryType.CODE_SNIPPET,
                tags=["python"]
            ),
            payload=PayloadLayer(content="Code content"),
            meta=MetaData(
                source_agent_id="test",
                user_id="u1",
                updated_at=datetime.now() - timedelta(days=3),
                confidence_score=0.85
            )
        )

    def test_index_only_rendering(self):
        """测试仅渲染 Index 层"""
        renderer = CompactContextRenderer(CompactRendererConfig())

        output = renderer.render([self.memory1, self.memory2])

        # 应该包含 Index 视图
        assert "<memory_index alias=" in output
        # 不应该包含完整内容
        assert "[完整内容]:" not in output
        # 应该包含摘要
        assert "**内容摘要**:" in output

    def test_index_template_structure(self):
        """测试 Index 模板结构"""
        renderer = CompactContextRenderer(CompactRendererConfig())

        output = renderer.render([self.memory1])

        assert "<memory_context>" in output
        assert "<memory_index alias=" in output
        assert "### Test Memory 1" in output
        assert "**类型**:" in output
        assert "**标签**:" in output
        assert "**内容摘要**:" in output

    def test_read_instruction_in_footer(self):
        """测试 Footer 中包含 READ 指令"""
        renderer = CompactContextRenderer(CompactRendererConfig())

        output = renderer.render([self.memory1])

        assert "⟪ READ" in output

    def test_empty_results(self):
        """测试空结果返回精简闭环提示"""
        from hivememory.engines.retrieval.renderer import _EMPTY_CONTEXT_NOTICE
        renderer = CompactContextRenderer(CompactRendererConfig())
        assert renderer.render([]) == _EMPTY_CONTEXT_NOTICE

    def test_budget_exhausted(self):
        """测试预算耗尽时返回精简闭环提示"""
        from hivememory.engines.retrieval.renderer import _EMPTY_CONTEXT_NOTICE
        config = CompactRendererConfig(
            max_memory_tokens=200,
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        assert output == _EMPTY_CONTEXT_NOTICE

    def test_summary_truncation(self):
        """测试摘要截断"""
        long_summary_memory = MemoryAtom(
            index=IndexLayer(
                title="Long Summary Memory",
                summary="This is a moderately long summary that exceeds the truncation limit. " * 3,
                memory_type=MemoryType.FACT,
                tags=["test"]
            ),
            payload=PayloadLayer(content="Content"),
            meta=MetaData(source_agent_id="test", user_id="u1")
        )

        config = CompactRendererConfig(
            max_memory_tokens=2000,
            index_max_summary_length=50,
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([long_summary_memory])

        # 摘要应该被截断并添加 "..."
        assert "..." in output


class TestCreateRenderer:
    """测试渲染器工厂函数"""

    def test_create_default_renderer(self):
        """测试默认创建 FullContextRenderer"""
        renderer = create_renderer(FullRendererConfig())
        assert isinstance(renderer, FullContextRenderer)

    def test_create_full_renderer(self):
        """测试创建 FullContextRenderer"""
        config = FullRendererConfig()
        renderer = create_renderer(config)
        assert isinstance(renderer, FullContextRenderer)

    def test_create_cascade_renderer(self):
        """测试创建 CascadeContextRenderer"""
        config = CascadeRendererConfig()
        renderer = create_renderer(config)
        assert isinstance(renderer, CascadeContextRenderer)

    def test_create_compact_renderer(self):
        """测试创建 CompactContextRenderer"""
        config = CompactRendererConfig()
        renderer = create_renderer(config)
        assert isinstance(renderer, CompactContextRenderer)

    def test_invalid_config_raises_error(self):
        """测试无效配置抛出错误"""
        with pytest.raises(ValueError, match="未知的渲染器配置类型"):
            create_renderer({"invalid": "config"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
