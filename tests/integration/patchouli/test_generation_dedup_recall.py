"""生成查重经真实向量召回命中已有记忆。

查重矩阵（相似度区间与内容一致性如何映射到判定）由查重器的单元测试维护；本文件只验证
真实 embedding 与真实 Qdrant 下，引擎的查重查询能召回内容相同的已有记忆，并把它判为
只触碰。查询与已有记忆的稠密向量文本形态不一致时，内容相同的草稿也到不了触碰所需的
相似度，会被当作相似内容新建后合并。

需要配置中的 Qdrant 服务与真实 embedding 模型；Qdrant 不可达时整文件跳过。
"""

import socket
from uuid import uuid4

import pytest
import pytest_asyncio

from hivememory.config.app import load_app_config
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    PayloadLayer,
    resolve_default_workspace_identity,
    system_actor_for_workspace,
)
from hivememory.engines.generation.deduplicator import MemoryDeduplicator
from hivememory.engines.generation.engine import MemoryGenerationEngine
from hivememory.engines.generation.interfaces import BaseMemoryExtractor
from hivememory.engines.generation.models import (
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationContext,
    GenerationRequest,
    GenerationTurn,
)
from hivememory.infrastructure.storage.vector_store import QdrantMemoryStore
from hivememory.memory_library import MidTermMemoryStore
from hivememory.memory_library.adapters.mid_term import QdrantStorageAdapter
from tests.helpers.memory import make_memory_metadata

_CONFIG = load_app_config()
_USER_ID = "dedup_recall_user"

_TITLE = "PyTorch 安装指南"
_SUMMARY = "PyTorch 深度学习框架的安装步骤和环境配置方法"
_CONTENT = """PyTorch 安装步骤：
1. 确认 Python 版本（推荐 3.8+）
2. 安装命令：pip install torch torchvision torchaudio
3. GPU 版本需要先安装 CUDA
4. 验证安装：import torch; print(torch.cuda.is_available())"""


def _qdrant_reachable() -> bool:
    storage = _CONFIG.patchouli.storage
    try:
        with socket.create_connection((storage.host, storage.port), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.real_infra,
    pytest.mark.skipif(not _qdrant_reachable(), reason="配置中的 Qdrant 服务不可达"),
]


class _FixedDraftExtractor(BaseMemoryExtractor):
    """测试边界外的提取器：话题提取总是给出同一份草稿。"""

    def __init__(self, draft: ExtractedMemoryDraft) -> None:
        self._draft = draft

    async def extract(self, transcript: str) -> list[ExtractedMemoryDraft]:
        return [self._draft]


@pytest_asyncio.fixture
async def mid_term():
    """真实 embedding 与 Qdrant 上的中期库；每个测试独占一个集合，结束时删除。"""
    storage = QdrantMemoryStore(
        qdrant_config=_CONFIG.patchouli.storage.model_copy(
            update={"collection_name": f"dedup_recall_{uuid4().hex}"}
        ),
        embedding_config=_CONFIG.shared.embedding.default,
    )
    await storage.create_collection()
    try:
        yield MidTermMemoryStore(QdrantStorageAdapter(storage))
    finally:
        await storage.client.delete_collection(collection_name=storage.collection_name)
        await storage.client.close()


@pytest.mark.asyncio
async def test_identical_draft_touches_existing_memory_through_real_recall(mid_term):
    """草稿与已有记忆的标题、摘要与正文相同时，结果只触碰这条记忆，不新建。"""
    existing = MemoryAtom(
        meta=make_memory_metadata(source_agent_id="a1", user_id=_USER_ID),
        index=IndexLayer(
            title=_TITLE,
            summary=_SUMMARY,
            tags=["PyTorch", "安装", "深度学习", "Python"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content=_CONTENT),
    )
    await mid_term.upsert(existing)
    draft = ExtractedMemoryDraft(
        title=_TITLE,
        summary=_SUMMARY,
        tags=["PyTorch", "安装", "深度学习"],
        memory_type="FACT",
        content=_CONTENT,
        confidence_score=0.85,
        has_value=True,
    )
    engine = MemoryGenerationEngine(
        mid_term=mid_term,
        extractor=_FixedDraftExtractor(draft),
        deduplicator=MemoryDeduplicator(config=_CONFIG.patchouli.generation.deduplicator),
    )
    belong_to = resolve_default_workspace_identity(_USER_ID)
    context = GenerationContext(
        turns=[
            GenerationTurn(
                user_message="PyTorch 怎么安装？",
                identity=ActorIdentity(user_id=_USER_ID, agent_id="a1"),
            )
        ]
    )

    outcomes = await engine.process(
        GenerationRequest(context=context),
        belong_to=belong_to,
        from_actor=system_actor_for_workspace(belong_to),
    )

    assert [(outcome.action, outcome.atom.id) for outcome in outcomes] == [
        (GenerationAction.TOUCH, existing.id)
    ]
