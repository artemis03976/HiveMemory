"""
生成引擎组件协作与集成测试

测试覆盖:
1. Extractor 与 Deduplicator 的协作 (Integration)
2. MemoryGenerationEngine 的完整处理流程 (Integration)
3. MemoryGenerationEngine 的单元逻辑 (Unit Logic)
   - 各种查重决策下的处理分支 (CREATE/TOUCH/UPDATE/DISCARD)
   - 异常处理

不测试：与外部服务（LLM、Qdrant）的真实网络交互（使用 Mock）
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
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
    StreamMessage,
    Identity
)
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationRequest,
    GenerationContext,
    GenerationTurn,
)
from hivememory.system.config import DeduplicatorConfig, ExtractorConfig
from hivememory.engines.generation import (
    LLMMemoryExtractor,
    MemoryDeduplicator,
    MemoryGenerationEngine,
)
from hivememory.engines.generation.interfaces import BaseMemoryExtractor, BaseDeduplicator
from hivememory.infrastructure.storage import QdrantMemoryStore


class TestExtractorAndDeduplicatorCollaboration:
    """测试 Extractor 与 Deduplicator 的协作 (Integration Level)"""

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

        extractor = LLMMemoryExtractor(llm_service=mock_llm, config=ExtractorConfig())
        deduplicator = MemoryDeduplicator(storage=Mock(), config=DeduplicatorConfig())

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


class TestMemoryGenerationEngineLogic:
    """测试 MemoryGenerationEngine 的内部逻辑 (Unit Level)"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.mock_storage = Mock(spec=QdrantMemoryStore)
        self.mock_extractor = Mock(spec=BaseMemoryExtractor)
        self.mock_deduplicator = Mock(spec=BaseDeduplicator)
        
        self.engine = MemoryGenerationEngine(
            storage=self.mock_storage,
            extractor=self.mock_extractor,
            deduplicator=self.mock_deduplicator
        )
        
        # 基础测试数据
        self.messages = [
            StreamMessage(message_type="user", content="Hi")
        ]
        
        self.draft = ExtractedMemoryDraft(
            title="Test",
            summary="This is a summary that is long enough",
            tags=["t1"],
            memory_type="FACT",
            content="Content",
            confidence_score=0.9,
            has_value=True
        )
        
        self.memory_atom = MemoryAtom(
            meta=MetaData(source_agent_id="a1", user_id="u1", session_id="s1", confidence_score=0.9),
            index=IndexLayer(title="Test", summary="This is a summary that is long enough", tags=["t1"], memory_type=MemoryType.FACT),
            payload=PayloadLayer(content="Content")
        )

    def _context_from_messages(self, messages: List[StreamMessage]) -> GenerationContext:
        turns = []
        for i in range(0, len(messages), 2):
            user_msg = messages[i] if i < len(messages) else None
            assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
            turns.append(
                GenerationTurn(
                    user_query=user_msg.content if user_msg else "",
                    assistant_final_text=assistant_msg.content if assistant_msg else "",
                    identity=(
                        assistant_msg.identity
                        if assistant_msg and assistant_msg.identity
                        else (user_msg.identity if user_msg and user_msg.identity else Identity())
                    ),
                )
            )
        return GenerationContext(turns=turns)

    def test_process_empty_messages(self):
        """测试空消息列表"""
        result = self.engine.process(GenerationRequest())
        assert result == []

    def test_process_extraction_fails(self):
        """测试提取失败"""
        self.mock_extractor.extract.return_value = None
        
        result = self.engine.process(GenerationRequest(context=self._context_from_messages(self.messages)))
        
        assert result == []
        self.mock_deduplicator.check_duplicate.assert_not_called()

    def test_process_create_new_memory(self):
        """测试创建新记忆流程"""
        self.mock_extractor.extract.return_value = self.draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.CREATE, None)
        
        result = self.engine.process(GenerationRequest(context=self._context_from_messages(self.messages)))
        
        assert len(result) == 1
        assert result[0].duplicate_decision == DuplicateDecision.CREATE
        assert result[0].atom is not None
        assert result[0].atom.index.title == "Test"

    def test_process_touch_existing_memory(self):
        """测试 TOUCH 现有记忆"""
        self.mock_extractor.extract.return_value = self.draft
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.TOUCH, self.memory_atom)
        
        result = self.engine.process(GenerationRequest(context=self._context_from_messages(self.messages)))
        
        assert len(result) == 1
        assert result[0].duplicate_decision == DuplicateDecision.TOUCH
        assert result[0].atom == self.memory_atom

        # 验证只更新访问信息，不重新插入
        self.mock_storage.update_access_info.assert_called_once_with(self.memory_atom.id)
        self.mock_storage.upsert_memory.assert_not_called()

    def test_process_update_memory(self):
        """测试 UPDATE 记忆演化"""
        self.mock_extractor.extract.return_value = self.draft
        
        merged_memory = self.memory_atom.model_copy()
        merged_memory.index.title = "Merged Title"
        
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.UPDATE, self.memory_atom)
        self.mock_deduplicator.merge_memory.return_value = merged_memory
        
        result = self.engine.process(GenerationRequest(context=self._context_from_messages(self.messages)))
        
        assert len(result) == 1
        assert result[0].duplicate_decision == DuplicateDecision.UPDATE
        assert result[0].atom is not None
        assert result[0].atom.index.title == "Merged Title"

        # 验证调用了合并
        self.mock_deduplicator.merge_memory.assert_called_once()

    def test_process_discard_memory(self):
        """测试 DISCARD 记忆"""
        self.mock_extractor.extract.return_value = self.draft
        # 模拟返回一个不在 (TOUCH, UPDATE, CREATE) 中的决策值，触发 else 分支 (DISCARD)
        self.mock_deduplicator.check_duplicate.return_value = (DuplicateDecision.DISCARD, None)
        
        result = self.engine.process(GenerationRequest(context=self._context_from_messages(self.messages)))
        assert len(result) == 1
        assert result[0].duplicate_decision == DuplicateDecision.DISCARD
        assert result[0].atom is None
        self.mock_storage.upsert_memory.assert_not_called()

    def test_draft_to_memory_conversion(self):
        """测试草稿转 MemoryAtom"""
        identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
        memory = self.engine._draft_to_memory(self.draft, identity)
        
        assert memory.index.title == "Test"
        assert memory.meta.user_id == "u1"
        assert memory.meta.source_agent_id == "a1"
        assert memory.index.memory_type == MemoryType.FACT

    def test_format_transcript(self):
        """测试对话格式化"""
        msgs = [
            StreamMessage(message_type="user", content="Hi"),
            StreamMessage(message_type="assistant", content="Hello")
        ]
        text = self.engine._render_transcript(
            GenerationRequest(context=self._context_from_messages(msgs))
        )

        assert "[User]: Hi" in text
        assert "[Assistant]: Hello" in text


class TestEngineComponentCoordination:
    """测试 MemoryGenerationEngine 对各组件的编排 (Integration Level Mocking)"""

    def test_engine_full_pipeline(self):
        """测试完整的处理流程：Extract -> Deduplicate"""
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

        engine = MemoryGenerationEngine(
            storage=Mock(),
            extractor=LLMMemoryExtractor(llm_service=mock_llm, config=ExtractorConfig()),
            deduplicator=mock_deduplicator,
        )

        messages = [
            StreamMessage(message_type="user", content="测试内容"),
            StreamMessage(message_type="assistant", content="测试回复"),
        ]

        context = GenerationContext(
            turns=[GenerationTurn(user_query="测试内容", assistant_final_text="测试回复", identity=Identity())]
        )
        result = engine.process(GenerationRequest(context=context))

        assert result is not None
        assert len(result) == 1
        assert result[0].duplicate_decision == DuplicateDecision.CREATE
        assert result[0].atom is not None
        assert result[0].atom.index.title == "测试记忆"
        
        # 验证组件被正确调用
        mock_llm.complete_with_retry.assert_called()
        mock_deduplicator.check_duplicate.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
