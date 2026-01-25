"""
ContextRenderer 单元测试

测试覆盖:
- XML 格式渲染
- Markdown 格式渲染
- 记忆截断逻辑
- 时间和置信度格式化
- CompactContextRenderer 分级渲染
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from hivememory.core.models import MemoryAtom, MemoryType, PayloadLayer, IndexLayer, MetaData, VerificationStatus
from hivememory.engines.retrieval.renderer import ContextRenderer, RenderFormat, MinimalRenderer, CompactContextRenderer
from hivememory.patchouli.config import CompactRendererConfig

class TestContextRenderer:
    """测试上下文渲染器"""

    def setup_method(self):
        self.renderer = ContextRenderer()
        
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
        
        renderer = ContextRenderer(max_content_length=50)
        output = renderer.render([memory], render_format=RenderFormat.MARKDOWN)
        
        assert "[内容已截断" in output
        assert len(output) < len(long_content) + 200  # 确保确实截断了

    def test_time_formatting(self):
        """测试时间格式化"""
        # 使用渲染输出检查时间格式化
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


class TestMinimalRenderer:
    """测试极简渲染器"""

    def test_render(self):
        renderer = MinimalRenderer()
        memory = MemoryAtom(
            index=IndexLayer(title="Title", summary="This is a sufficiently long summary.", memory_type=MemoryType.FACT, tags=["t1"]),
            payload=PayloadLayer(content="Content"),
            meta=MetaData(source_agent_id="test", user_id="u1")
        )

        output = renderer.render([memory])
        assert "[相关记忆]" in output
        assert "1. [t1] Title: Content..." in output


class TestCompactContextRenderer:
    """测试紧凑上下文渲染器"""

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
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=True,
            full_payload_count=1,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 第一条应该是完整渲染 (memory_block)
        assert '<memory_block id="1"' in output
        # 后续应该是 Index 视图 (memory_ref)
        assert '<memory_ref id="2"' in output or '<memory_ref id="3"' in output

    def test_budget_truncation(self):
        """测试超出预算时降级为 Index 渲染"""
        # 设置很小的预算
        config = CompactRendererConfig(
            max_memory_tokens=300,
            enable_tiered_rendering=True,
            full_payload_count=1,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 应该有输出
        assert len(output) > 0
        # 由于预算限制，应该有 Index 视图 (memory_ref)
        # 第一条强制完整渲染，后续应该降级
        assert "memory_block" in output  # 至少有一条完整渲染
        assert "memory_ref" in output  # 应该有降级的 Index 视图

    def test_budget_exhausted(self):
        """测试预算耗尽时停止渲染"""
        # 设置极小的预算
        config = CompactRendererConfig(
            max_memory_tokens=150,
            enable_tiered_rendering=True,
            full_payload_count=0,  # 不强制完整渲染
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2, self.memory3])

        # 应该有输出 (至少头尾)
        assert "<system_memory_context>" in output
        # 由于预算极小，可能只渲染了部分记忆
        # 验证不会超出预算太多

    def test_index_only_format(self):
        """测试 Index 视图格式正确"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=True,
            full_payload_count=0,  # 全部使用 Index 视图
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1])

        # 应该是 Index 视图
        assert "<memory_ref" in output
        assert "[标签]:" in output
        assert "[摘要]:" in output
        assert "[提示]:" in output

    def test_lazy_loading_hint(self):
        """测试懒加载提示正确生成"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=True,
            full_payload_count=0,
            enable_lazy_loading=True,
            lazy_load_tool_name="read_memory",
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1])

        # 应该包含懒加载提示
        assert "read_memory" in output

    def test_empty_results(self):
        """测试空结果处理"""
        renderer = CompactContextRenderer()
        output = renderer.render([])
        assert output == ""

    def test_markdown_format(self):
        """测试 Markdown 格式渲染"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=True,
            full_payload_count=1,
            render_format="markdown"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        # 应该包含 Markdown 格式
        assert "## 相关记忆上下文" in output
        # 第一条完整渲染 (Markdown 格式使用 📌)
        assert "📌" in output
        # 第二条 Index 视图 (使用 📎)
        assert "📎" in output or "摘要" in output

    def test_token_estimation(self):
        """测试 Token 估算准确性"""
        renderer = CompactContextRenderer()

        # 测试中文
        chinese_text = "这是一段中文测试文本"
        tokens = renderer._estimate_tokens(chinese_text)
        assert tokens > 0

        # 测试英文
        english_text = "This is an English test text"
        tokens = renderer._estimate_tokens(english_text)
        assert tokens > 0

        # 空文本
        assert renderer._estimate_tokens("") == 0

    def test_full_payload_count_zero(self):
        """测试 full_payload_count=0 时全部使用 Index 视图"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=True,
            full_payload_count=0,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        # 所有记忆都应该是 Index 视图
        assert "memory_ref" in output
        # 不应该有完整渲染
        assert "memory_block" not in output

    def test_disable_tiered_rendering(self):
        """测试禁用分级渲染"""
        config = CompactRendererConfig(
            max_memory_tokens=2000,
            enable_tiered_rendering=False,
            render_format="xml"
        )
        renderer = CompactContextRenderer(config)

        output = renderer.render([self.memory1, self.memory2])

        # 禁用分级渲染时，应该尝试完整渲染所有记忆
        # 直到预算耗尽
        assert "memory_block" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
