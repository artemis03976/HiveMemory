"""
Renderer 单元测试

测试覆盖:
- FullContextRenderer: XML/Markdown 格式渲染、截断逻辑
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
from hivememory.patchouli.config import (
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

    def test_render_xml(self):
        """测试 XML 渲染"""
        output = self.renderer.render([self.memory1, self.memory2], render_format=RenderFormat.XML)

        assert "<system_memory_context>" in output
        assert '<memory_block id="1" type="FACT">' in output
        assert "#test" in output
        assert "#unit" in output
        assert "This is the content of memory 1." in output
        assert '<memory_block id="2" type="CODE_SNIPPET">' in output
        assert "</system_memory_context>" in output

    def test_render_markdown(self):
        """测试 Markdown 渲染"""
        output = self.renderer.render([self.memory1], render_format=RenderFormat.MARKDOWN)

        assert "## 相关记忆上下文" in output
        assert "### 📌 Test Memory 1" in output
        assert "- **类型**: `FACT`" in output
        assert "`test`" in output

    def test_empty_results(self):
        """测试空结果"""
        assert self.renderer.render([]) == ""

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
        output = renderer.render([memory], render_format=RenderFormat.MARKDOWN)

        assert "[内容已截断" in output
        assert len(output) < len(long_content) + 200

    def test_time_formatting(self):
        """测试时间格式化"""
        # 2小时前
        output = self.renderer.render([self.memory1], render_format=RenderFormat.XML)
        assert "小时前" in output

        # 5天前
        output = self.renderer.render([self.memory2], render_format=RenderFormat.XML)
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
        output = self.renderer.render([old_memory], render_format=RenderFormat.XML)
        assert "个月前" in output

    def test_confidence_formatting(self):
        """测试置信度格式化"""
        from hivememory.utils.memory_atom_renderer import MemoryAtomRenderer

        # 高置信度
        assert "✓" in MemoryAtomRenderer._format_confidence(self.memory1)
        # 中置信度
        assert "~" in MemoryAtomRenderer._format_confidence(self.memory2)

    def test_with_config(self):
        """测试使用配置初始化"""
        config = FullRendererConfig(
            render_format="markdown",
            max_tokens=1000,
            max_content_length=100,
            show_artifacts=False,
            stale_days=30
        )
        renderer = FullContextRenderer(config)

        assert renderer.render_format == RenderFormat.MARKDOWN
        assert renderer.max_tokens == 1000
        assert renderer.max_content_length == 100


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
            render_format="xml"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 第一条应该是完整渲染 (memory_block)
        assert '<memory_block id="1"' in output
        # 后续应该是 Index 视图 (memory_ref)
        assert '<memory_ref id="2"' in output or '<memory_ref id="3"' in output

    def test_budget_truncation(self):
        """测试超出预算时降级为 Index 渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=300,
            full_payload_count=1,
            render_format="xml"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        assert len(output) > 0
        # 由于预算限制，应该有 Index 视图 (memory_ref)
        assert "memory_ref" in output

    def test_budget_exhausted(self):
        """测试预算耗尽时停止渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=150,
            full_payload_count=0,
            render_format="xml"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        assert "<system_memory_context>" in output

    def test_lazy_loading_hint(self):
        """测试懒加载提示正确生成"""
        config = CascadeRendererConfig(
            max_memory_tokens=2000,
            full_payload_count=0,
            enable_lazy_loading=True,
            lazy_load_tool_name="read_memory",
            render_format="xml"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1])

        assert "read_memory" in output

    def test_empty_results(self):
        """测试空结果处理"""
        renderer = CascadeContextRenderer(CascadeRendererConfig())
        output = renderer.render([])
        assert output == ""

    def test_markdown_format(self):
        """测试 Markdown 格式渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=2000,
            full_payload_count=1,
            render_format="markdown"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        assert "## 相关记忆上下文" in output
        # 第一条完整渲染
        assert "📌" in output or "Test Memory 1" in output
        # 第二条 Index 视图
        assert "摘要" in output

    def test_full_payload_count_multiple(self):
        """测试 full_payload_count > 1 时多条完整渲染"""
        config = CascadeRendererConfig(
            max_memory_tokens=5000,
            full_payload_count=2,
            render_format="xml"
        )
        renderer = CascadeContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 前两条应该是完整渲染
        assert '<memory_block id="1"' in output
        assert '<memory_block id="2"' in output
        # 第三条应该是 Index 视图
        assert '<memory_ref id="3"' in output


class TestCompactContextRenderer:
    """测试紧凑上下文渲染器 (仅 Index 层)"""

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

    def test_index_only_rendering(self):
        """测试仅渲染 Index 层"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        # 所有记忆都应该是 Index 视图
        assert "memory_ref" in output
        # 不应该有完整渲染
        assert "memory_block" not in output

    def test_index_format_xml(self):
        """测试 Index 视图 XML 格式正确"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1])

        assert "<memory_ref" in output
        assert "[标签]:" in output
        assert "[摘要]:" in output
        assert "[提示]:" in output

    def test_index_format_markdown(self):
        """测试 Index 视图 Markdown 格式正确"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            render_format="markdown"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1])

        assert "## 相关记忆上下文" in output
        assert "摘要" in output

    def test_lazy_loading_enabled_by_default(self):
        """测试懒加载默认启用"""
        renderer = CompactContextRenderer(CompactRendererConfig())

        output = renderer.render([self.memory1])

        # CompactRendererConfig 默认 enable_lazy_loading=True
        assert "read_memory" in output

    def test_lazy_loading_hint(self):
        """测试懒加载提示正确生成"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_lazy_loading=True,
            lazy_load_tool_name="get_memory_detail",
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1])

        assert "get_memory_detail" in output

    def test_empty_results(self):
        """测试空结果处理"""
        renderer = CompactContextRenderer(CompactRendererConfig())
        output = renderer.render([])
        assert output == ""

    def test_budget_exhausted(self):
        """测试预算耗尽时停止渲染"""
        config = CompactRendererConfig(
            max_memory_tokens=150,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        assert "<system_memory_context>" in output

    def test_summary_truncation(self):
        """测试摘要截断"""
        # 创建一个摘要长度超过 index_max_summary_length 但不超过 IndexLayer 限制的记忆
        long_summary_memory = MemoryAtom(
            index=IndexLayer(
                title="Long Summary Memory",
                summary="This is a moderately long summary that exceeds the truncation limit. " * 3,  # ~200 chars
                memory_type=MemoryType.FACT,
                tags=["test"]
            ),
            payload=PayloadLayer(content="Content"),
            meta=MetaData(source_agent_id="test", user_id="u1")
        )

        config = CompactRendererConfig(
            max_memory_tokens=2000,
            index_max_summary_length=50,  # 设置较小的截断限制
            render_format="xml"
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
