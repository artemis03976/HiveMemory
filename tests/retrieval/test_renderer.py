"""
ContextRenderer 单元测试

测试覆盖:
- XML 格式渲染
- Markdown 格式渲染
- 记忆截断逻辑
- 时间和置信度格式化
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from hivememory.core.models import MemoryAtom, MemoryType, PayloadLayer, IndexLayer, MetaData, VerificationStatus
from hivememory.retrieval.renderer import ContextRenderer, RenderFormat, MinimalRenderer

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
        # 2小时前
        assert "小时前" in self.renderer._format_time_ago(datetime.now() - timedelta(hours=2))
        # 5天前
        assert "天前" in self.renderer._format_time_ago(datetime.now() - timedelta(days=5))
        # 40天前
        assert "个月前" in self.renderer._format_time_ago(datetime.now() - timedelta(days=40))

    def test_confidence_formatting(self):
        """测试置信度格式化"""
        # 高置信度
        assert "✓" in self.renderer._format_confidence(self.memory1)
        # 中置信度
        assert "~" in self.renderer._format_confidence(self.memory2)


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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
