"""
生成引擎组件协作测试

测试生成引擎内部各组件之间的协作：
- Extractor 与 Deduplicator 的协作
- Gating 机制与 Extractor 的交互
- GenerationOrchestrator 对各组件的编排

不测试：与外部服务（LLM、Qdrant）的交互
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from hivememory.core.models import (
    MemoryAtom,
    MetaData,
    IndexLayer,
    PayloadLayer,
    MemoryType,
)
from hivememory.engines.generation.models import (
    ConversationMessage,
    ExtractedMemoryDraft,
)
from hivememory.engines.generation import (
    LLMMemoryExtractor,
    MemoryDeduplicator,
    LLMAssistedGater,
    MemoryGenerationOrchestrator,
    DuplicateDecision,
)


class TestExtractorAndDeduplicatorCollaboration:
    """测试 Extractor 与 Deduplicator 的协作"""

    def test_extractor_calls_deduplicator(self):
        """测试 Extractor 输出可被 Deduplicator 处理"""
        # 创建 Mock LLM 服务
        mock_llm = Mock()
        mock_llm.complete_with_retry = Mock(return_value='''
            {
                "title": "Python 函数",
                "summary": "一个测试函数",
                "tags": ["python", "test"],
                "memory_type": "CODE_SNIPPET",
                "content": "def test(): pass",
                "confidence_score": 0.9,
                "has_value": true
            }
        ''')

        extractor = LLMMemoryExtractor(llm_service=mock_llm)
        deduplicator = MemoryDeduplicator(storage=Mock())

        messages = [
            ConversationMessage(role="user", content="写一个Python函数"),
            ConversationMessage(role="assistant", content="```python\ndef test(): pass\n```"),
        ]

        # 提取记忆
        draft = extractor.extract(
            transcript="transcript",
            metadata={"user_id": "test_user"}
        )

        assert draft is not None
        assert draft.title == "Python 函数"

        # 应用去重检查
        decision, existing = deduplicator.check_duplicate(draft)

        # 验证去重逻辑被调用 (Mock storage 会返回空，所以应该是 CREATE)
        assert decision == DuplicateDecision.CREATE


class TestGatingAndExtractorCollaboration:
    """测试 Gating 与 Extractor 的协作"""

    def test_extractor_skips_gated_content(self):
        """测试 Extractor 跳过被 Gating 的内容"""
        mock_gating = Mock()
        # Mock evaluate returning False (not worth saving)
        mock_gating.evaluate = Mock(return_value=False)

        mock_llm = Mock()
        # Should not be called
        mock_llm.complete_with_retry = Mock()

        orchestrator = MemoryGenerationOrchestrator(
            storage=Mock(),
            gater=mock_gating,
            extractor=LLMMemoryExtractor(llm_service=mock_llm),
            deduplicator=Mock()
        )

        messages = [
            ConversationMessage(role="user", content="你好"),
            ConversationMessage(role="assistant", content="你好！"),
        ]

        result = orchestrator.process(messages)

        # Gating 应该被调用
        mock_gating.evaluate.assert_called_once()

        # 由于被 Gating，不应该调用 LLM
        mock_llm.complete_with_retry.assert_not_called()
        
        # 返回空结果
        assert result == []


class TestOrchestratorComponentCoordination:
    """测试 GenerationOrchestrator 对各组件的编排"""

    def test_orchestrator_full_pipeline(self):
        """测试完整的处理流程：Gating -> Extract -> Deduplicate"""
        mock_gating = Mock()
        mock_gating.evaluate = Mock(return_value=True)

        mock_llm = Mock()
        mock_llm.complete_with_retry = Mock(return_value='''
            {
                "title": "测试记忆",
                "summary": "这是一个测试用的摘要信息，长度必须超过十个字符",
                "tags": ["test"],
                "content": "测试内容",
                "memory_type": "FACT",
                "confidence_score": 0.9,
                "has_value": true
            }
        ''')

        mock_deduplicator = Mock()
        mock_deduplicator.check_duplicate = Mock(return_value=(DuplicateDecision.CREATE, None))

        orchestrator = MemoryGenerationOrchestrator(
            storage=Mock(),
            gater=mock_gating,
            extractor=LLMMemoryExtractor(llm_service=mock_llm),
            deduplicator=mock_deduplicator,
        )
        # Mock internal draft_to_memory or ensure it works
        # _draft_to_memory uses MemoryType(draft.memory_type)

        messages = [
            ConversationMessage(role="user", content="测试内容"),
            ConversationMessage(role="assistant", content="测试回复"),
        ]

        result = orchestrator.process(messages)

        assert result is not None
        assert len(result) == 1
        assert result[0].index.title == "测试记忆"
        
        # 验证组件被正确调用
        mock_gating.evaluate.assert_called_once()
        mock_llm.complete_with_retry.assert_called()
        mock_deduplicator.check_duplicate.assert_called()


    def test_orchestrator_handles_empty_messages(self):
        """测试处理空消息列表"""
        orchestrator = MemoryGenerationOrchestrator(
            storage=Mock(),
            gater=Mock(),
            extractor=Mock(),
            deduplicator=Mock(),
        )

        result = orchestrator.process([])

        # 应该返回空结果而不是报错
        assert result == []

class TestMemoryConversion:
    """测试 ExtractedMemory 到 MemoryAtom 的转换"""

    def test_extracted_memory_to_atom_conversion(self):
        """测试 ExtractedMemory 转换为 MemoryAtom"""
        extracted = ExtractedMemoryDraft(
            title="测试记忆",
            summary="这是一个测试用的摘要信息，长度必须超过十个字符",
            content="测试内容",
            tags=["test", "demo"],
            memory_type="FACT",
            confidence_score=0.9,
            has_value=True,
        )

        # 转换为 MemoryAtom
        # We can use the orchestrator's helper method or replicate the logic
        orchestrator = MemoryGenerationOrchestrator(storage=Mock())
        atom = orchestrator._draft_to_memory(
            draft=extracted,
            user_id="test_user",
            agent_id="test_agent",
            session_id="test_session"
        )

        assert atom.index.title == "测试记忆"
        assert atom.index.memory_type == MemoryType.FACT
        assert atom.meta.confidence_score == 0.9
        assert atom.meta.user_id == "test_user"
        assert atom.meta.source_agent_id == "test_agent"



# Helper class for mocking
class MagicCtx:
    """Helper class for creating mock context managers"""
    def __init__(self, **kwargs):
        self.data = kwargs

    def __enter__(self):
        return MagicMock(**self.data)

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
