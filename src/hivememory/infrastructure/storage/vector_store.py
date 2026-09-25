"""
Qdrant 向量存储层封装

提供:
- 集合管理(创建、删除)
- 记忆原子的 CRUD 操作
- 混合检索(向量 + 元数据过滤)
"""

import logging
from collections.abc import Iterable
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from hivememory.core.models import MemoryAtom, WorkspaceMemoryKey
from hivememory.core.mtp.exceptions import (
    StorageOfflineError,
    StorageReadError,
    StorageWriteError,
    SystemFault,
)
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget
from hivememory.engines.retrieval.memory_codec import MemoryDecodeError, decode_memory_payload
from hivememory.infrastructure.embedding import get_bge_m3_service
from hivememory.infrastructure.storage.qdrant_client import (
    create_async_qdrant_client,
    wait_for_qdrant_ready,
)
from hivememory.system.config import EmbeddingConfig, QdrantConfig

logger = logging.getLogger(__name__)

_compiler = MemoryCompiler()

# 连接类异常表示存储不可达，与读写本身的失败分开归类。
_OFFLINE_ERRORS = (ConnectionError, TimeoutError, OSError)


def _storage_error(exc: Exception, *, operation: str, write: bool) -> SystemFault:
    """把存储异常归类为结构化错误：连接类为离线，其余按读/写区分。

    失败必须传播给调用方：读取失败不能伪装成"没有记忆"，写入/删除失败不能
    伪装成已生效（A2-P M0.3）。
    """
    logger.error("Memory 存储操作失败: %s: %s", operation, exc, exc_info=True)
    if isinstance(exc, _OFFLINE_ERRORS):
        return StorageOfflineError(cause=exc)
    if write:
        return StorageWriteError(cause=exc)
    return StorageReadError(cause=exc)


def _key_of(memory: MemoryAtom) -> WorkspaceMemoryKey:
    return WorkspaceMemoryKey(workspace_identity=memory.workspace_identity, memory_id=memory.id)


class QdrantMemoryStore:
    """
    Qdrant 向量存储管理器 (async-native)

    职责:
    1. 管理向量集合生命周期
    2. 记忆原子的存储与检索
    3. Embedding 向量生成
    """

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        embedding_config: EmbeddingConfig,
    ):
        """
        初始化存储管理器

        Args:
            qdrant_config: Qdrant 配置
            embedding_config: Embedding 配置

        注意: 直接实例化配置类会自动从环境变量读取值
        """
        # 使用默认配置（直接实例化会读取环境变量）
        self.qdrant_config = qdrant_config
        self.embedding_config = embedding_config

        self.client = create_async_qdrant_client(self.qdrant_config)

        logger.info("加载 BGE-M3 Embedding 服务")

        bge_config = self.embedding_config
        if "bge-m3" not in bge_config.model_name.lower():
            logger.info("当前 Embedding 配置非 BGE-M3，自动调整模型名称以适配存储层")
            bge_config = bge_config.model_copy(update={"model_name": "Xenova/bge-m3"})

        self.embedding_service = get_bge_m3_service(config=bge_config)

        self.collection_name = self.qdrant_config.collection_name
        self.vector_dimension = self.qdrant_config.vector_dimension

    async def ensure_ready(self) -> None:
        await wait_for_qdrant_ready(
            self.client,
            timeout_seconds=self.qdrant_config.startup_timeout_seconds,
        )
        await self.create_collection(recreate=False)

    async def create_collection(self, recreate: bool = False) -> None:
        try:
            collections = (await self.client.get_collections()).collections
            collection_exists = any(col.name == self.collection_name for col in collections)

            if collection_exists:
                if recreate:
                    logger.warning(f"删除已存在的集合: {self.collection_name}")
                    await self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"集合已存在且有稀疏向量配置: {self.collection_name}")
                    return

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_text": VectorParams(
                        size=self.vector_dimension,
                        distance=getattr(Distance, self.qdrant_config.distance_metric.upper()),
                    ),
                },
                sparse_vectors_config={"sparse_text": SparseVectorParams(modifier=Modifier.IDF)},
                on_disk_payload=self.qdrant_config.on_disk_payload,
            )

            logger.info(f"✓ 成功创建集合: {self.collection_name} (Dense + Sparse)")

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    async def upsert_memory(
        self, memory: MemoryAtom, use_sparse: bool = True, recompute_vectors: bool = True
    ) -> None:
        """
        插入或更新记忆原子

        Args:
            memory: 记忆原子对象
            use_sparse: 是否同时存储稀疏向量
            recompute_vectors: 是否重算 embedding；``False`` 用于 embedding
                输入未变化的完整内容提交（如仅修改 ``payload.agent_config``），
                通过整份 payload 局部更新保留既有向量

        Raises:
            StorageOfflineError / StorageWriteError: 写入失败
        """
        point_id = self._point_id(_key_of(memory))
        try:
            if not recompute_vectors:
                # embedding 输入未变化：set_payload 只替换 payload 顶层键，向量原样保留。
                await self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=memory.to_qdrant_payload(),
                    points=[point_id],
                )
                logger.debug(f"✓ 已保留向量并替换 payload: {memory.id}")
                return

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=self._encode_vectors(memory, use_sparse=use_sparse),
                        payload=memory.to_qdrant_payload(),
                    )
                ],
            )
            logger.debug(f"✓ 成功存储记忆: {memory.id} - {memory.index.title}")
        except Exception as e:
            raise _storage_error(e, operation="upsert_memory", write=True) from e

    def _encode_vectors(self, memory: MemoryAtom, *, use_sparse: bool) -> dict[str, Any]:
        """编译 embedding 输入并生成命名向量：dense 必有，sparse 为 BM25 文本。"""
        dense_text = _compiler.compile(memory, MemoryCompileTarget.DENSE_EMBEDDING).text
        if not use_sparse:
            return {"dense_text": self.embedding_service.encode(dense_texts=dense_text)}
        sparse_text = _compiler.compile(memory, MemoryCompileTarget.SPARSE_EMBEDDING).text
        vectors = self.embedding_service.encode(dense_texts=dense_text, sparse_texts=sparse_text)
        return {
            "dense_text": vectors["dense"],
            "sparse_text": Document(text=vectors["sparse_text"], model="qdrant/bm25"),
        }

    async def patch_memory_payload(
        self,
        key: WorkspaceMemoryKey,
        *,
        lifecycle: dict[str, Any] | None = None,
        access_policy: dict[str, Any] | None = None,
    ) -> None:
        """对单个点做受限局部 payload 更新，向量与未提交字段不动。

        仅允许两个嵌套键：``meta.lifecycle``（整块替换）与
        ``meta.access_policy``（整体替换）；由 adapter 的字段白名单保证调用
        方无法触达其他路径。
        """
        try:
            point_id = self._point_id(key)
            if lifecycle is not None:
                await self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=lifecycle,
                    points=[point_id],
                    key="meta.lifecycle",
                )
            if access_policy is not None:
                await self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=access_policy,
                    points=[point_id],
                    key="meta.access_policy",
                )
        except Exception as e:
            raise _storage_error(e, operation="patch_memory_payload", write=True) from e

    async def get_memory(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        """按复合键读取单条 Memory；记录无法安全解码时整体失败。"""
        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._point_id(key)],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            return decode_memory_payload(points[0].payload or {})
        except Exception as e:
            raise _storage_error(e, operation="get_memory", write=False) from e

    async def get_memory_by_alias(
        self,
        alias: str,
        *,
        query_filter: Filter,
    ) -> MemoryAtom | None:
        """
        根据别名精确匹配检索记忆 (L2 Cold Lookup, MTP Section 2.3.2)

        使用 scroll API + FieldCondition 精确匹配 index.alias 字段。

        Args:
            alias: 语义化别名 (e.g. "code_quicksort_impl")
            query_filter: 已构造的过滤条件，必须包含 Workspace ownership 边界

        Returns:
            MemoryAtom 对象，未找到返回 None
        """
        alias_filter = FieldCondition(key="index.alias", match=MatchValue(value=alias))
        try:
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[query_filter, alias_filter]),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            return decode_memory_payload(points[0].payload or {})
        except Exception as e:
            raise _storage_error(e, operation=f"get_memory_by_alias({alias})", write=False) from e

    async def search_memories(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Filter | None = None,
        mode: str = "dense",
    ) -> list[dict[str, Any]]:
        """
        语义检索记忆 (支持稠密和稀疏向量检索)

        Args:
            query_text: 查询文本
            top_k: 返回Top K结果
            score_threshold: 最低相似度阈值（仅稠密检索使用）
            filters: 已构造的元数据过滤条件，必须包含 Workspace ownership 边界
            mode: 检索模式，"dense" 使用稠密向量，"sparse" 使用稀疏向量

        Returns:
            检索结果列表: [{"memory": MemoryAtom, "score": float, "id": point_id}, ...]
        """
        try:
            if mode == "sparse":
                query: Any = Document(text=query_text, model="qdrant/bm25")
                using, threshold = "sparse_text", None
            else:
                query = self.embedding_service.encode(dense_texts=query_text)
                using, threshold = "dense_text", score_threshold
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query,
                using=using,
                query_filter=filters,
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
            )
        except Exception as e:
            raise _storage_error(e, operation="search_memories", write=False) from e
        logger.debug("✓ %s 检索到 %d 条记忆", using, len(response.points))

        return [
            {"memory": memory, "score": hit.score, "id": hit.id}
            for hit, memory in self._decode_points(response.points, operation="search_memories")
        ]

    async def delete_memory(self, key: WorkspaceMemoryKey) -> bool:
        """删除复合键对应的点；失败以结构化错误传播，不返回 False 掩盖。"""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[self._point_id(key)],
            )
        except Exception as e:
            raise _storage_error(e, operation="delete_memory", write=True) from e
        logger.debug(f"✓ 成功删除记忆: {key.memory_id}")
        return True

    async def get_all_memories(
        self,
        *,
        filters: Filter,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        """
        按过滤条件 scroll 记忆（不做相似度排序）。

        Args:
            filters: 已构造的过滤条件，必须包含 Workspace ownership 边界
            limit: 最多返回多少条（默认100）

        Returns:
            MemoryAtom 列表
        """
        points = await self._scroll(filters, limit=limit, operation="get_all_memories")
        return [memory for _, memory in self._decode_points(points, operation="get_all_memories")]

    async def get_all_memories_for_maintenance(
        self,
        *,
        limit: int = 10000,
    ) -> list[MemoryAtom]:
        """进程级维护遍历（不限 Workspace）；无法解码的记录跳过并告警。"""
        points = await self._scroll(None, limit=limit, operation="get_all_memories_for_maintenance")
        return [
            memory
            for _, memory in self._decode_points(
                points, operation="get_all_memories_for_maintenance"
            )
        ]

    # ========== 内部辅助方法 ==========

    async def _scroll(self, filters: Filter | None, *, limit: int, operation: str) -> list[Any]:
        """读取一页 payload（不含向量）；失败以结构化读取错误传播。"""
        try:
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filters,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            raise _storage_error(e, operation=operation, write=False) from e
        return points

    @staticmethod
    def _decode_points(points: Iterable[Any], *, operation: str) -> list[tuple[Any, MemoryAtom]]:
        """逐点解码；无法安全解码的记录 fail closed 跳过并告警，不中断整批。"""
        decoded = []
        for point in points:
            try:
                decoded.append((point, decode_memory_payload(point.payload or {})))
            except MemoryDecodeError as exc:
                logger.warning(
                    "%s 跳过无法安全解码的 Memory: point_id=%s, error=%s",
                    operation,
                    point.id,
                    exc,
                )
        return decoded

    @staticmethod
    def _point_id(key: WorkspaceMemoryKey) -> str:
        """将复合资源键稳定映射为 Qdrant 支持的 UUID point id。"""
        workspace = key.workspace_identity
        canonical = (
            f"hivememory-memory:{workspace.owner_user_id}:"
            f"{workspace.workspace_id}:{key.memory_id}"
        )
        return str(uuid5(NAMESPACE_URL, canonical))
