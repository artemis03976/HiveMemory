"""
HiveMemory Generation Module E2E Tests

测试 MemoryGenerationEngine 的核心逻辑。

测试组：
    - Group 1: 记忆提取测试 (Extraction)
    - Group 2: 去重决策测试 (Deduplication Logic)
    - Group 3: 记忆合并测试 (Merger)
    - Group 4: Schema 验证测试

运行方式：
    pytest tests/components/test_generation_e2e.py -v

核心原则：
    - 使用真实的 LiteLLMService（librarian 配置）
    - 使用真实的 QdrantMemoryStore
    - 聚焦记忆提取、去重、合并机制

作者: HiveMemory Team
版本: 1.0.0
"""

import sys
import os
from pathlib import Path

from tests.helpers.memory import make_memory_metadata

# UTF-8 编码配置 (Windows 兼容性)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

# ========== 日志配置（必须在导入其他模块之前） ==========

import logging

# 配置根日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# 关闭第三方库的 INFO/DEBUG 日志
_log_levels_to_disable = {
    "FlagEmbedding": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "httpx": logging.WARNING,
    "litellm": logging.WARNING,
}

for logger_name, level in _log_levels_to_disable.items():
    logging.getLogger(logger_name).setLevel(level)

# ========== 其他导入 ==========

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]
from rich.console import Console
from rich.panel import Panel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 核心模型
from hivememory.core.models import (
    ActorIdentity,
    StreamMessage,
    StreamMessageType,
    MemoryAtom,
    IndexLayer,
    PayloadLayer,
    MemoryType,
)

# Generation 模块组件
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.extractor import LLMMemoryExtractor
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
    MergeResult,
)

# 配置
from hivememory.system.config import (
    load_app_config,
    DeduplicatorConfig,
)

# 基础设施
from hivememory.infrastructure.llm.litellm_service import get_librarian_llm_service
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore

# 导入测试数据
from tests.fixtures.generation_test_data import (
    EXTRACTION_TEST_CASES,
    DEDUPLICATION_TEST_CASES,
    MERGE_TEST_CASES,
    SCHEMA_VALIDATION_CASES,
    EXISTING_MEMORY_DATA,
)

# 导入 conftest 中的辅助函数
from tests.conftest import print_test_result

console = Console(force_terminal=True, legacy_windows=False)

# ========== 全局测试状态 ==========

_shared_storage: Optional[QdrantMemoryStore] = None
_shared_extractor: Optional[LLMMemoryExtractor] = None
_shared_deduplicator: Optional[MemoryDeduplicator] = None
_shared_engine: Optional[MemoryGenerationEngine] = None
_test_collection_name: str = "hivememory_generation_test"


def setup_test_env() -> MemoryGenerationEngine:
    """
    初始化测试环境

    创建真实的 MemoryGenerationEngine 及其依赖组件。

    Returns:
        MemoryGenerationEngine: 配置好的生成引擎实例
    """
    global _shared_storage, _shared_extractor, _shared_deduplicator, _shared_engine

    if _shared_engine is not None:
        return _shared_engine

    console.print(Panel("[bold cyan]初始化 Generation E2E 测试环境[/bold cyan]"))

    # 加载配置
    app_config = load_app_config()

    # 1. 创建 LLM 服务（使用 librarian 配置）
    llm_config = app_config.get_librarian_llm_config()
    console.print(f"[dim]LLM 模型: {llm_config.model}[/dim]")
    llm_service = get_librarian_llm_service(config=llm_config)

    # 2. 创建 QdrantMemoryStore（使用测试集合）
    qdrant_config = app_config.qdrant.model_copy(update={"collection_name": _test_collection_name})
    console.print(f"[dim]Qdrant 集合: {qdrant_config.collection_name}[/dim]")
    _shared_storage = QdrantMemoryStore(
        qdrant_config=qdrant_config,
        embedding_config=app_config.embedding.default,
    )

    # 确保测试集合存在
    asyncio.run(_shared_storage.create_collection(recreate=False))

    # 3. 创建 LLMMemoryExtractor
    _shared_extractor = LLMMemoryExtractor(
        config=app_config.generation.extractor,
        llm_service=llm_service,
    )

    # 4. 创建 MemoryDeduplicator
    dedup_config = app_config.generation.deduplicator
    _shared_deduplicator = MemoryDeduplicator(
        storage=_shared_storage,
        config=dedup_config,
    )

    # 5. 创建 MemoryGenerationEngine
    _shared_engine = MemoryGenerationEngine(
        storage=_shared_storage,
        extractor=_shared_extractor,
        deduplicator=_shared_deduplicator,
    )

    console.print("[green]Generation E2E 测试环境初始化完成[/green]")

    return _shared_engine


def get_shared_engine() -> MemoryGenerationEngine:
    """获取共享的 Generation Engine 实例"""
    global _shared_engine
    if _shared_engine is None:
        return setup_test_env()
    return _shared_engine


def get_shared_storage() -> QdrantMemoryStore:
    """获取共享的 Storage 实例"""
    global _shared_storage
    if _shared_storage is None:
        setup_test_env()
    return _shared_storage


def get_shared_extractor() -> LLMMemoryExtractor:
    """获取共享的 Extractor 实例"""
    global _shared_extractor
    if _shared_extractor is None:
        setup_test_env()
    return _shared_extractor


def get_shared_deduplicator() -> MemoryDeduplicator:
    """获取共享的 Deduplicator 实例"""
    global _shared_deduplicator
    if _shared_deduplicator is None:
        setup_test_env()
    return _shared_deduplicator


def reset_test_env() -> None:
    """
    重置测试环境

    清空测试集合中的数据，确保测试隔离。
    """
    global _shared_storage

    if _shared_storage is not None:
        try:
            # 删除并重建测试集合
            asyncio.run(_shared_storage.create_collection(recreate=True))
            console.print("[dim]测试集合已重置[/dim]")
        except Exception as e:
            console.print(f"[yellow]重置测试集合失败: {e}[/yellow]")


def create_test_identity(prefix: str = "test") -> ActorIdentity:
    """创建测试用的 ActorIdentity"""
    return ActorIdentity(
        user_id=f"{prefix}_user_{uuid.uuid4().hex[:8]}",
        agent_id=f"{prefix}_agent",
        session_id=f"{prefix}_session_{uuid.uuid4().hex[:8]}",
    )


def create_stream_messages(
    messages: List[Dict[str, str]],
    identity: ActorIdentity
) -> List[StreamMessage]:
    """将测试数据转换为 StreamMessage 列表"""
    role_mapping = {
        "user": StreamMessageType.USER,
        "assistant": StreamMessageType.ASSISTANT,
        "system": StreamMessageType.SYSTEM,
    }
    
    return [
        StreamMessage(
            message_type=role_mapping.get(msg["role"], StreamMessageType.USER),
            content=msg["content"],
            identity=identity,
        )
        for msg in messages
    ]


def create_generation_context(
    messages: List[Dict[str, str]],
    identity: ActorIdentity,
) -> GenerationContext:
    """将测试消息转换为 generation 主路径使用的 GenerationContext。"""
    turns: List[GenerationTurn] = []
    for i in range(0, len(messages), 2):
        user_msg = messages[i] if i < len(messages) else None
        assistant_msg = messages[i + 1] if i + 1 < len(messages) else None
        turns.append(
            GenerationTurn(
                user_query=user_msg["content"] if user_msg else "",
                assistant_final_text=assistant_msg["content"] if assistant_msg else "",
                identity=identity,
            )
        )
    return GenerationContext(turns=turns)


def create_memory_from_data(data: Dict[str, Any], identity: ActorIdentity) -> MemoryAtom:
    """从测试数据创建 MemoryAtom"""
    try:
        mem_type = MemoryType(data["memory_type"])
    except ValueError:
        mem_type = MemoryType.FACT

    # 处理 ID：如果 data["id"] 是有效 UUID 则使用，否则生成新的
    try:
        if "id" in data:
            memory_id = uuid.UUID(data["id"])
        else:
            memory_id = uuid.uuid4()
    except ValueError:
        # 如果提供的 ID 不是有效 UUID 格式，生成一个新的
        memory_id = uuid.uuid4()

    return MemoryAtom(
        id=memory_id,
        meta=make_memory_metadata(
            source_agent_id=identity.agent_id,
            user_id=identity.user_id,
            session_id=identity.session_id,
            confidence_score=data.get("confidence_score", 0.8),
        ),
        index=IndexLayer(
            title=data["title"],
            summary=data["summary"],
            tags=data["tags"],
            memory_type=mem_type,
        ),
        payload=PayloadLayer(
            content=data["content"],
        ),
    )


def create_draft_from_data(data: Dict[str, Any]) -> ExtractedMemoryDraft:
    """从测试数据创建 ExtractedMemoryDraft"""
    return ExtractedMemoryDraft(
        title=data["title"],
        summary=data["summary"],
        tags=data["tags"],
        memory_type=data["memory_type"],
        content=data["content"],
        confidence_score=data.get("confidence_score", 0.8),
        has_value=data.get("has_value", True),
    )


# ========== Group 1: 记忆提取测试 (Extraction) ==========

class TestMemoryExtraction:
    """
    Group 1: 记忆提取测试

    验证 LLMMemoryExtractor 的提取能力。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.extractor = get_shared_extractor()
        self.identity = create_test_identity("extraction")

    def test_standard_info_extraction(self):
        """
        GEN-EXT-001: 标准信息提取

        验证点：
        - LLM 能从事实性对话中提取 MemoryDraft
        - 包含 title/content/summary
        - has_value=True
        """
        test_case = EXTRACTION_TEST_CASES[0]  # GEN-EXT-001
        assert test_case["id"] == "GEN-EXT-001"

        # 格式化对话
        transcript = self._format_transcript(test_case["messages"])

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": self.identity.user_id,
                "agent_id": self.identity.agent_id,
                "session_id": self.identity.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 验证结果
        success = True
        error_msg = None

        if draft is None:
            success = False
            error_msg = "提取结果为 None"
        elif not draft.has_value:
            success = False
            error_msg = f"has_value={draft.has_value}, 预期 True"
        else:
            # 检查标题是否包含关键词
            title_check = any(
                keyword.lower() in draft.title.lower()
                for keyword in test_case.get("expected_title_contains", [])
            )
            if not title_check and test_case.get("expected_title_contains"):
                success = False
                error_msg = f"标题 '{draft.title}' 不包含预期关键词"

        print_test_result(console, "GEN-EXT-001: 标准信息提取", success, error_msg)

        if draft:
            console.print(f"    [dim]标题: {draft.title}[/dim]")
            console.print(f"    [dim]类型: {draft.memory_type}[/dim]")
            console.print(f"    [dim]has_value: {draft.has_value}[/dim]")
            console.print(f"    [dim]置信度: {draft.confidence_score:.2f}[/dim]")

        assert success, error_msg

    def test_noise_filtering(self):
        """
        GEN-EXT-002: 噪音过滤

        验证点：
        - 无营养闲聊返回 has_value=False 或 None
        """
        test_case = EXTRACTION_TEST_CASES[1]  # GEN-EXT-002
        assert test_case["id"] == "GEN-EXT-002"

        # 格式化对话
        transcript = self._format_transcript(test_case["messages"])

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": self.identity.user_id,
                "agent_id": self.identity.agent_id,
                "session_id": self.identity.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 验证结果：应该返回 None 或 has_value=False
        success = (draft is None) or (not draft.has_value)

        print_test_result(console, "GEN-EXT-002: 噪音过滤", success)

        if draft:
            console.print(f"    [dim]has_value: {draft.has_value} (预期: False)[/dim]")
        else:
            console.print(f"    [dim]返回 None (符合预期)[/dim]")

        assert success, f"噪音对话应返回 None 或 has_value=False，实际: {draft}"

    def test_complex_structure_extraction(self):
        """
        GEN-EXT-003: 复杂结构提取 - 代码片段

        验证点：
        - 代码片段完整提取
        - Tags 包含技术关键词
        - memory_type 为 CODE_SNIPPET
        """
        test_case = EXTRACTION_TEST_CASES[2]  # GEN-EXT-003
        assert test_case["id"] == "GEN-EXT-003"

        # 格式化对话
        transcript = self._format_transcript(test_case["messages"])

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": self.identity.user_id,
                "agent_id": self.identity.agent_id,
                "session_id": self.identity.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 验证结果
        success = True
        error_msg = None

        if draft is None:
            success = False
            error_msg = "提取结果为 None"
        elif not draft.has_value:
            success = False
            error_msg = f"has_value={draft.has_value}, 预期 True"
        else:
            # 检查内容是否包含代码关键词
            content_keywords = test_case.get("expected_content_contains", [])
            missing_keywords = [k for k in content_keywords if k not in draft.content]
            if missing_keywords:
                success = False
                error_msg = f"内容缺少关键词 {missing_keywords}"

            # 检查标签
            expected_tags = test_case.get("expected_tags_any", [])
            tag_match = any(
                tag.lower() in [t.lower() for t in draft.tags]
                for tag in expected_tags
            )
            if expected_tags and not tag_match:
                success = False
                error_msg = f"标签 {draft.tags} 不包含预期标签 {expected_tags}"

        print_test_result(console, "GEN-EXT-003: 复杂结构提取", success, error_msg)

        if draft:
            console.print(f"    [dim]标题: {draft.title}[/dim]")
            console.print(f"    [dim]类型: {draft.memory_type}[/dim]")
            console.print(f"    [dim]标签: {draft.tags}[/dim]")
            console.print(f"    [dim]内容长度: {len(draft.content)} 字符[/dim]")

        assert success, error_msg

    def test_user_preference_extraction(self):
        """
        GEN-EXT-004: 用户偏好提取

        验证点：
        - 能识别用户偏好信息
        - memory_type 为 USER_PROFILE
        """
        test_case = EXTRACTION_TEST_CASES[3]  # GEN-EXT-004
        assert test_case["id"] == "GEN-EXT-004"

        # 格式化对话
        transcript = self._format_transcript(test_case["messages"])

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": self.identity.user_id,
                "agent_id": self.identity.agent_id,
                "session_id": self.identity.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 验证结果
        success = draft is not None and draft.has_value

        print_test_result(console, "GEN-EXT-004: 用户偏好提取", success)

        if draft:
            console.print(f"    [dim]标题: {draft.title}[/dim]")
            console.print(f"    [dim]类型: {draft.memory_type}[/dim]")
            console.print(f"    [dim]标签: {draft.tags}[/dim]")

        assert success, "用户偏好对话应成功提取"

    def _format_transcript(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话为文本"""
        lines = []
        for msg in messages:
            role_display = {
                "user": "👤 User",
                "assistant": "🤖 Assistant",
                "system": "⚙️ System"
            }.get(msg["role"], msg["role"])
            lines.append(f"{role_display}: {msg['content']}")
        return "\n".join(lines)


# ========== Group 2: 去重决策测试 (Deduplication Logic) ==========

class TestDeduplicationLogic:
    """
    Group 2: 去重决策测试

    验证 MemoryDeduplicator 的决策逻辑。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.deduplicator = get_shared_deduplicator()
        self.storage = get_shared_storage()
        self.identity = create_test_identity("dedup")
        # 重置测试环境
        reset_test_env()

    def test_decision_create_new_memory(self):
        """
        GEN-DED-001: 决策 CREATE - 新记忆

        验证点：
        - 无相似记忆时返回 DuplicateDecision.CREATE
        """
        test_case = DEDUPLICATION_TEST_CASES[0]  # GEN-DED-001
        assert test_case["id"] == "GEN-DED-001"

        # 先插入一条现有记忆
        existing_key = test_case["existing_memory_key"]
        existing_data = EXISTING_MEMORY_DATA[existing_key]
        existing_memory = create_memory_from_data(existing_data, self.identity)
        asyncio.run(self.storage.upsert_memory(existing_memory))

        # 创建新草稿（与现有记忆不相似）
        draft = create_draft_from_data(test_case["draft_data"])

        # 调用查重
        decision, found_memory = self.deduplicator.check_duplicate(draft)

        # 验证结果
        expected_decision = test_case["expected_decision"]
        success = decision.value == expected_decision.lower()

        print_test_result(console, "GEN-DED-001: 决策 CREATE", success)
        console.print(f"    [dim]决策结果: {decision.value} (预期: {expected_decision.lower()})[/dim]")
        console.print(f"    [dim]新草稿标题: {draft.title}[/dim]")
        console.print(f"    [dim]现有记忆标题: {existing_memory.index.title}[/dim]")

        assert success, f"预期决策 {expected_decision}，实际 {decision.value}"

    def test_decision_touch_duplicate(self):
        """
        GEN-DED-002: 决策 TOUCH - 完全重复

        验证点：
        - 高相似度 + 内容一致时返回 TOUCH
        """
        test_case = DEDUPLICATION_TEST_CASES[1]  # GEN-DED-002
        assert test_case["id"] == "GEN-DED-002"

        # 先插入现有记忆
        existing_key = test_case["existing_memory_key"]
        existing_data = EXISTING_MEMORY_DATA[existing_key]
        existing_memory = create_memory_from_data(existing_data, self.identity)
        asyncio.run(self.storage.upsert_memory(existing_memory))

        # 创建几乎相同的草稿
        draft = create_draft_from_data(test_case["draft_data"])

        # 调用查重
        decision, found_memory = self.deduplicator.check_duplicate(draft)

        # 验证结果
        expected_decision = test_case["expected_decision"]
        # TOUCH 或 UPDATE 都可以接受（取决于内容相似度计算）
        success = decision.value in [expected_decision.lower(), "update"]

        print_test_result(console, "GEN-DED-002: 决策 TOUCH/UPDATE", success)
        console.print(f"    [dim]决策结果: {decision.value} (预期: {expected_decision.lower()} 或 update)[/dim]")
        if found_memory:
            console.print(f"    [dim]匹配记忆: {found_memory.index.title}[/dim]")

        assert success, f"预期决策 {expected_decision} 或 UPDATE，实际 {decision.value}"

    def test_decision_update_evolution(self):
        """
        GEN-DED-003: 决策 UPDATE - 知识演化

        验证点：
        - 中等相似度或内容有增量时返回 UPDATE
        """
        test_case = DEDUPLICATION_TEST_CASES[2]  # GEN-DED-003
        assert test_case["id"] == "GEN-DED-003"

        # 先插入现有记忆
        existing_key = test_case["existing_memory_key"]
        existing_data = EXISTING_MEMORY_DATA[existing_key]
        existing_memory = create_memory_from_data(existing_data, self.identity)
        asyncio.run(self.storage.upsert_memory(existing_memory))

        # 创建有增量的草稿
        draft = create_draft_from_data(test_case["draft_data"])

        # 调用查重
        decision, found_memory = self.deduplicator.check_duplicate(draft)

        # 验证结果
        expected_decision = test_case["expected_decision"]
        success = decision.value == expected_decision.lower()

        print_test_result(console, "GEN-DED-003: 决策 UPDATE", success)
        console.print(f"    [dim]决策结果: {decision.value} (预期: {expected_decision.lower()})[/dim]")
        console.print(f"    [dim]新草稿: {draft.title}[/dim]")
        if found_memory:
            console.print(f"    [dim]匹配记忆: {found_memory.index.title}[/dim]")

        assert success, f"预期决策 {expected_decision}，实际 {decision.value}"


# ========== Group 3: 记忆合并测试 (Merger) ==========

class TestMemoryMerger:
    """
    Group 3: dedup UPDATE 测试

    验证 GenerationEngine 在 dedup UPDATE 下覆盖当前 head，
    历史版本由 MemoryVersionArtifact 链路负责。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.engine = get_shared_engine()
        self.identity = create_test_identity("merger")

    def test_dedup_update_replaces_content_head(self):
        """
        GEN-MRG-001: dedup UPDATE 覆盖当前内容

        验证点：
        - payload.content 等于新草稿内容
        - 旧内容不会拼接污染当前 head
        """
        test_case = MERGE_TEST_CASES[0]  # GEN-MRG-001
        assert test_case["id"] == "GEN-MRG-001"

        # 创建现有记忆
        existing_data = test_case["existing_memory"]
        existing_memory = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id=self.identity.agent_id,
                user_id=self.identity.user_id,
                session_id=self.identity.session_id,
                confidence_score=existing_data["confidence_score"],
            ),
            index=IndexLayer(
                title=existing_data["title"],
                summary=existing_data["summary"],
                tags=existing_data["tags"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(
                content=existing_data["content"],
            ),
        )

        # 创建新草稿
        new_draft = create_draft_from_data(test_case["new_draft"])

        old_content = existing_memory.payload.content
        result = self.engine._apply_update(
            existing_memory,
            MergeResult(
                new_content=new_draft.content,
                changelog=f"Dedup update: {new_draft.summary[:120]}",
            ),
            dedup_draft=new_draft,
        )
        merged = result[0].atom

        success = merged.payload.content == new_draft.content and old_content not in merged.payload.content
        print_test_result(console, "GEN-MRG-001: dedup UPDATE 覆盖当前内容", success)
        console.print(f"    [dim]更新后内容长度: {len(merged.payload.content)} 字符[/dim]")

        assert success, "dedup UPDATE 不应把旧内容追加进当前 head"

    def test_dedup_update_refreshes_tags(self):
        """
        GEN-MRG-002: 标签并集刷新

        验证点：
        - 更新后 Tags 为两者并集（去重，最多5个）
        """
        test_case = MERGE_TEST_CASES[1]  # GEN-MRG-002
        assert test_case["id"] == "GEN-MRG-002"

        # 创建现有记忆
        existing_data = test_case["existing_memory"]
        existing_memory = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id=self.identity.agent_id,
                user_id=self.identity.user_id,
                session_id=self.identity.session_id,
                confidence_score=existing_data["confidence_score"],
            ),
            index=IndexLayer(
                title=existing_data["title"],
                summary=existing_data["summary"],
                tags=existing_data["tags"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(
                content=existing_data["content"],
            ),
        )

        # 创建新草稿
        new_draft = create_draft_from_data(test_case["new_draft"])

        # 执行 dedup UPDATE
        merged = self.engine._apply_update(
            existing_memory,
            MergeResult(
                new_content=new_draft.content,
                changelog=f"Dedup update: {new_draft.summary[:120]}",
            ),
            dedup_draft=new_draft,
        )[0].atom

        # 验证结果
        merged_tags = set(merged.index.tags)
        expected_superset = set(test_case.get("expected_tags_superset", []))
        max_tags = test_case.get("expected_max_tags", 5)

        # 检查标签数量限制
        tags_count_ok = len(merged.index.tags) <= max_tags

        # 检查是否包含部分预期标签
        common_tags = merged_tags & expected_superset
        tags_content_ok = len(common_tags) >= 2  # 至少包含2个预期标签

        success = tags_count_ok and tags_content_ok

        print_test_result(console, "GEN-MRG-002: 标签并集刷新", success)
        console.print(f"    [dim]原标签: {existing_data['tags']}[/dim]")
        console.print(f"    [dim]新标签: {new_draft.tags}[/dim]")
        console.print(f"    [dim]更新后: {merged.index.tags}[/dim]")
        console.print(f"    [dim]标签数量: {len(merged.index.tags)} (最大: {max_tags})[/dim]")

        assert success, f"标签合并不符合预期"

    def test_dedup_update_replaces_summary(self):
        """
        GEN-MRG-003: 摘要覆盖策略

        验证点：
        - 更新后 summary 直接采用草稿摘要，不按长度取舍
        """
        test_case = MERGE_TEST_CASES[2]  # GEN-MRG-003
        assert test_case["id"] == "GEN-MRG-003"

        # 创建现有记忆（短摘要）
        existing_data = test_case["existing_memory"]
        existing_memory = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id=self.identity.agent_id,
                user_id=self.identity.user_id,
                session_id=self.identity.session_id,
                confidence_score=existing_data["confidence_score"],
            ),
            index=IndexLayer(
                title=existing_data["title"],
                summary=existing_data["summary"],
                tags=existing_data["tags"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(
                content=existing_data["content"],
            ),
        )

        # 创建新草稿（长摘要）
        new_draft = create_draft_from_data(test_case["new_draft"])

        # 执行 dedup UPDATE
        merged = self.engine._apply_update(
            existing_memory,
            MergeResult(
                new_content=new_draft.content,
                changelog=f"Dedup update: {new_draft.summary[:120]}",
            ),
            dedup_draft=new_draft,
        )[0].atom

        success = merged.index.summary == new_draft.summary

        print_test_result(console, "GEN-MRG-003: 摘要覆盖策略", success)
        console.print(f"    [dim]原摘要长度: {len(existing_data['summary'])}[/dim]")
        console.print(f"    [dim]新摘要长度: {len(new_draft.summary)}[/dim]")
        console.print(f"    [dim]更新后长度: {len(merged.index.summary)}[/dim]")

        assert success, "dedup UPDATE 应直接采用草稿摘要"


# ========== Group 4: Schema 验证测试 ==========

class TestSchemaValidation:
    """
    Group 4: Schema 验证测试

    验证生成的 MemoryAtom 符合 Schema 规范。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.engine = get_shared_engine()
        self.deduplicator = get_shared_deduplicator()
        self.extractor = get_shared_extractor()
        self.identity = create_test_identity("schema")

    def test_json_schema_compliance(self):
        """
        GEN-SCH-001: JSON Schema 合规

        验证点：
        - 生成的 MemoryAtom 包含所有必需字段
        """
        test_case = SCHEMA_VALIDATION_CASES[0]  # GEN-SCH-001
        assert test_case["id"] == "GEN-SCH-001"

        # 格式化对话
        transcript = self._format_transcript(test_case["messages"])

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": self.identity.user_id,
                "agent_id": self.identity.agent_id,
                "session_id": self.identity.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 验证 draft 不为空
        assert draft is not None and draft.has_value, "Schema 合规测试需要非空提取结果"

        # 转换为 MemoryAtom
        memory = self._draft_to_memory(draft)

        # 验证必需字段
        required_fields = test_case.get("required_fields", [])
        missing_fields = []

        for field_path in required_fields:
            if not self._check_field_exists(memory, field_path):
                missing_fields.append(field_path)

        success = len(missing_fields) == 0

        print_test_result(console, "GEN-SCH-001: JSON Schema 合规", success)
        console.print(f"    [dim]检查字段数: {len(required_fields)}[/dim]")
        if missing_fields:
            console.print(f"    [red]缺失字段: {missing_fields}[/red]")
        else:
            console.print(f"    [dim]所有必需字段均存在[/dim]")

        # 打印 MemoryAtom 结构
        console.print(f"    [dim]ID: {memory.id}[/dim]")
        console.print(f"    [dim]标题: {memory.index.title}[/dim]")
        console.print(f"    [dim]类型: {memory.index.memory_type}[/dim]")

        assert success, f"缺失必需字段: {missing_fields}"

    def test_update_confidence_reset(self):
        """
        GEN-SCH-002: 更新置信度重置

        验证点：
        - 统一 UPDATE primitive 应将更新后置信度置为 1.0
        """
        test_case = SCHEMA_VALIDATION_CASES[1]  # GEN-SCH-002
        assert test_case["id"] == "GEN-SCH-002"

        old_confidence = test_case["existing_confidence"]
        new_confidence = test_case["new_confidence"]

        # 创建现有记忆
        existing_memory = MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id=self.identity.agent_id,
                user_id=self.identity.user_id,
                session_id=self.identity.session_id,
                confidence_score=old_confidence,
            ),
            index=IndexLayer(
                title="测试记忆",
                summary="用于测试置信度计算的详细摘要信息",
                tags=["测试"],
                memory_type=MemoryType.FACT,
            ),
            payload=PayloadLayer(
                content="原始内容",
            ),
        )

        # 创建新草稿
        new_draft = ExtractedMemoryDraft(
            title="测试记忆更新",
            summary="用于测试置信度计算的更新",
            tags=["测试", "更新"],
            memory_type="FACT",
            content="新内容",
            confidence_score=new_confidence,
            has_value=True,
        )

        # 执行统一 UPDATE primitive
        merged = self.engine._apply_update(
            existing_memory,
            MergeResult(
                new_content=new_draft.content,
                changelog=f"Dedup update: {new_draft.summary[:120]}",
            ),
            dedup_draft=new_draft,
        )[0].atom

        # 验证置信度计算
        actual_confidence = merged.meta.confidence_score
        success = actual_confidence == 1.0

        print_test_result(console, "GEN-SCH-002: 更新置信度重置", success)
        console.print(f"    [dim]原置信度: {old_confidence}[/dim]")
        console.print(f"    [dim]新置信度: {new_confidence}[/dim]")
        console.print(f"    [dim]预期结果: 1.0[/dim]")
        console.print(f"    [dim]实际结果: {actual_confidence:.4f}[/dim]")

        assert success, f"更新置信度不符合预期: 预期 1.0，实际 {actual_confidence}"

    def _format_transcript(self, messages: List[Dict[str, str]]) -> str:
        """格式化对话为文本"""
        lines = []
        for msg in messages:
            role_display = {
                "user": "👤 User",
                "assistant": "🤖 Assistant",
                "system": "⚙️ System"
            }.get(msg["role"], msg["role"])
            lines.append(f"{role_display}: {msg['content']}")
        return "\n".join(lines)

    def _draft_to_memory(self, draft: ExtractedMemoryDraft) -> MemoryAtom:
        """将草稿转换为 MemoryAtom"""
        try:
            mem_type = MemoryType(draft.memory_type)
        except ValueError:
            mem_type = MemoryType.FACT

        return MemoryAtom(
            meta=make_memory_metadata(
                source_agent_id=self.identity.agent_id,
                user_id=self.identity.user_id,
                session_id=self.identity.session_id,
                confidence_score=draft.confidence_score,
            ),
            index=IndexLayer(
                title=draft.title,
                summary=draft.summary,
                tags=draft.tags,
                memory_type=mem_type,
            ),
            payload=PayloadLayer(
                content=draft.content,
            ),
        )

    def _check_field_exists(self, obj: Any, field_path: str) -> bool:
        """
        检查对象中是否存在指定路径的字段

        Args:
            obj: 要检查的对象
            field_path: 字段路径，如 "meta.user_id"

        Returns:
            bool: 字段是否存在且不为 None
        """
        parts = field_path.split(".")
        current = obj

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
                if current is None:
                    return False
            else:
                return False

        return True


# ========== 端到端流程测试 ==========

class TestEndToEndFlow:
    """
    端到端流程测试

    验证完整的记忆生成流程。
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前初始化"""
        self.engine = get_shared_engine()
        self.storage = get_shared_storage()
        self.identity = create_test_identity("e2e")
        # 重置测试环境
        reset_test_env()

    def test_full_generation_flow(self):
        """
        E2E-001: 完整生成流程

        验证点：
        - 从对话到记忆存储的完整流程
        """
        # 准备测试数据
        test_case = EXTRACTION_TEST_CASES[0]  # 使用标准信息提取测试数据

        # 调用引擎处理
        context = create_generation_context(test_case["messages"], self.identity)
        memories = asyncio.run(self.engine.process(GenerationRequest(context=context)))

        # 验证结果
        success = len(memories) > 0

        print_test_result(console, "E2E-001: 完整生成流程", success)
        console.print(f"    [dim]输入消息数: {len(test_case['messages'])}[/dim]")
        console.print(f"    [dim]生成记忆数: {len(memories)}[/dim]")

        if memories:
            memory = memories[0]
            console.print(f"    [dim]记忆标题: {memory.index.title}[/dim]")
            console.print(f"    [dim]记忆类型: {memory.index.memory_type}[/dim]")
            console.print(f"    [dim]记忆ID: {memory.id}[/dim]")

        assert success, "完整生成流程应产生至少一条记忆"

    def test_noise_rejection_flow(self):
        """
        E2E-002: 噪音拒绝流程

        验证点：
        - 无价值对话不产生记忆
        """
        # 准备噪音测试数据
        test_case = EXTRACTION_TEST_CASES[1]  # 噪音过滤测试数据

        # 调用引擎处理
        context = create_generation_context(test_case["messages"], self.identity)
        memories = asyncio.run(self.engine.process(GenerationRequest(context=context)))

        # 验证结果：噪音对话不应产生记忆
        success = len(memories) == 0

        print_test_result(console, "E2E-002: 噪音拒绝流程", success)
        console.print(f"    [dim]输入消息数: {len(test_case['messages'])}[/dim]")
        console.print(f"    [dim]生成记忆数: {len(memories)} (预期: 0)[/dim]")

        assert success, "噪音对话不应产生记忆"


# ========== 主函数 ==========

def run_all_tests():
    """运行所有测试（用于直接执行）"""
    console.print(Panel("[bold magenta]Generation E2E Tests[/bold magenta]", expand=False))

    # 初始化环境
    setup_test_env()

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
