"""
Qdrant 向量存储层封装

提供:
- 集合管理(创建、删除)
- 记忆原子的 CRUD 操作
- 混合检索(向量 + 元数据过滤)
- 批量操作
"""

from typing import List, Optional, Dict, Any, Union
from uuid import UUID
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SparseVectorParams,
    SparseVector,
    Document,
    Modifier,
)

from hivememory.core.models import MemoryAtom, IndexLayer
from hivememory.system.config import QdrantConfig, EmbeddingConfig
from hivememory.infrastructure.embedding import get_bge_m3_service
from hivememory.utils.memory_atom_renderer import MemoryAtomRenderer

logger = logging.getLogger(__name__)


class QdrantMemoryStore:
    """
    Qdrant 向量存储管理器

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

        # 初始化 Qdrant 客户端
        # 本地部署时 api_key 为 None，不传递给客户端
        client_kwargs = {
            "host": self.qdrant_config.host,
            "port": self.qdrant_config.port,
            "grpc_port": self.qdrant_config.grpc_port,
            "timeout": 60,
        }

        # 只有在 api_key 有值时才添加
        if self.qdrant_config.api_key and self.qdrant_config.api_key.strip():
            client_kwargs["api_key"] = self.qdrant_config.api_key

        self.client = QdrantClient(**client_kwargs)

        # 部分环境下 Qdrant HTTP 端点可能被网关拦截返回 502，
        # 这里探测一次并自动回退到 gRPC 以保证读写链路可用。
        try:
            self.client.get_collections()
        except Exception as e:
            err_text = str(e)
            if "Unexpected Response: 502" in err_text:
                logger.warning(
                    "Qdrant HTTP endpoint returned 502, fallback to gRPC client. "
                    "host=%s port=%s grpc_port=%s",
                    self.qdrant_config.host,
                    self.qdrant_config.port,
                    self.qdrant_config.grpc_port,
                )
                grpc_kwargs = {
                    **client_kwargs,
                    "prefer_grpc": True,
                    "check_compatibility": False,
                }
                self.client = QdrantClient(**grpc_kwargs)
            else:
                raise

        # 初始化 BGE-M3 Embedding 服务 (支持 Dense + Sparse)
        logger.info(f"加载 BGE-M3 Embedding 服务")
        
        # 确保配置指向 BGE-M3 模型
        bge_config = self.embedding_config
        if "bge-m3" not in bge_config.model_name.lower():
            logger.info("当前 Embedding 配置非 BGE-M3，自动调整模型名称以适配存储层")
            bge_config = bge_config.model_copy(update={"model_name": "Xenova/bge-m3"})
            
        self.embedding_service = get_bge_m3_service(config=bge_config)

        self.collection_name = self.qdrant_config.collection_name
        self.vector_dimension = self.qdrant_config.vector_dimension

    def create_collection(self, recreate: bool = False) -> None:
        """
        创建向量集合 (包含稠密和稀疏向量配置)

        Args:
            recreate: 如果集合已存在，是否删除并重建

        Raises:
            Exception: 创建失败时抛出
        """
        try:
            # 检查集合是否存在
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if collection_exists:
                if recreate:
                    logger.warning(f"删除已存在的集合: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"集合已存在且有稀疏向量配置: {self.collection_name}")
                    return

            # 创建新集合 (支持稠密和稀疏向量)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_text": VectorParams(
                        size=self.vector_dimension,
                        distance=getattr(Distance, self.qdrant_config.distance_metric.upper()),
                    ),
                },
                sparse_vectors_config={
                    "sparse_text": SparseVectorParams(modifier=Modifier.IDF)
                },
                on_disk_payload=self.qdrant_config.on_disk_payload,
            )

            logger.info(f"✓ 成功创建集合: {self.collection_name} (Dense + Sparse)")

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def upsert_memory(
        self,
        memory: MemoryAtom,
        use_sparse: bool = True,
        force_regenerate: bool = False
    ) -> None:
        """
        插入或更新记忆原子

        Args:
            memory: 记忆原子对象
            use_sparse: 是否同时存储稀疏向量
            force_regenerate: 是否强制重新生成向量

        Raises:
            Exception: 操作失败时抛出
        """
        try:
            if use_sparse:
                # 生成混合向量 (稠密 + 稀疏)，使用不同的输入文本
                dense_text = MemoryAtomRenderer.for_dense_embedding(memory)
                sparse_context = MemoryAtomRenderer.for_sparse_embedding(memory)
                vectors = self.embedding_service.encode(
                    dense_texts=dense_text,
                    sparse_texts=sparse_context
                )

                # 构建 Qdrant Point - dense 向量 + BM25 文本
                sparse_text = vectors["sparse_text"]
                point = PointStruct(
                    id=str(memory.id),
                    vector={
                        "dense_text": vectors["dense"],
                        "sparse_text": Document(text=sparse_text, model="qdrant/bm25"),
                    },
                    payload=memory.to_qdrant_payload(),
                )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
                logger.debug(f"✓ 成功存储记忆 (Dense+Sparse): {memory.id} - {memory.index.title}")
            else:
                # 仅使用稠密向量
                embedding_text = MemoryAtomRenderer.for_dense_embedding(memory)
                embedding = self.embedding_service.encode(dense_texts=embedding_text)

                # 构建 Qdrant Point - 使用命名向量格式以保持一致性
                point = PointStruct(
                    id=str(memory.id),
                    vector={
                        "dense_text": embedding,  # 命名稠密向量
                    },
                    payload=memory.to_qdrant_payload(),
                )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
                logger.debug(f"✓ 成功存储记忆 (Dense): {memory.id} - {memory.index.title}")

        except Exception as e:
            logger.error(f"存储记忆失败: {e}")
            raise

    def get_memory(self, memory_id: UUID) -> Optional[MemoryAtom]:
        """
        根据 ID 获取记忆

        Args:
            memory_id: 记忆UUID

        Returns:
            MemoryAtom 对象，不存在则返回 None
        """
        from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
        from hivememory.patchouli.mtp.exceptions import StorageOfflineError, StorageReadError

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(memory_id)],
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                return None

            # 重构 MemoryAtom
            payload = points[0].payload
            return self._payload_to_memory(payload)

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during get_memory: {e}")
            raise StorageOfflineError("Memory storage is unreachable.") from e
        except (UnexpectedResponse, ResponseHandlingException) as e:
            logger.error(f"Storage error during get_memory: {e}")
            raise StorageReadError("Memory storage returned an error.") from e
        except Exception as e:
            logger.error(f"Unexpected storage error in get_memory: {e}", exc_info=True)
            raise StorageReadError("Memory storage encountered an unexpected error.") from e

    def get_memory_by_alias(
        self,
        alias: str,
        user_id: Optional[str] = None,
    ) -> Optional[MemoryAtom]:
        """
        根据别名精确匹配检索记忆 (L2 Cold Lookup, MTP Section 2.3.2)

        使用 scroll API + FieldCondition 精确匹配 index.alias 字段。

        Args:
            alias: 语义化别名 (e.g. "code_quicksort_impl")
            user_id: 可选的用户 ID 过滤

        Returns:
            MemoryAtom 对象，未找到返回 None
        """
        from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
        from hivememory.patchouli.mtp.exceptions import StorageOfflineError, StorageReadError

        try:
            filters: Dict[str, Any] = {"index.alias": alias}
            if user_id:
                filters["meta.user_id"] = user_id

            filter_obj = self._build_filter(filters)

            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            points = scroll_result[0]
            if not points:
                return None

            return self._payload_to_memory(points[0].payload)

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during get_memory_by_alias (alias={alias}): {e}")
            raise StorageOfflineError("Memory storage is unreachable.") from e
        except (UnexpectedResponse, ResponseHandlingException) as e:
            logger.error(f"Storage error during get_memory_by_alias (alias={alias}): {e}")
            raise StorageReadError("Memory storage returned an error.") from e
        except Exception as e:
            logger.error(f"Unexpected storage error in get_memory_by_alias (alias={alias}): {e}", exc_info=True)
            raise StorageReadError("Memory storage encountered an unexpected error.") from e

    def search_memories(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Union[Dict[str, Any], Filter]] = None,
        mode: str = "dense",
    ) -> List[Dict[str, Any]]:
        """
        语义检索记忆 (支持稠密和稀疏向量检索)

        Args:
            query_text: 查询文本
            top_k: 返回Top K结果
            score_threshold: 最低相似度阈值
            filters: 元数据过滤条件, 如 {"memory_type": "CODE_SNIPPET", "user_id": "123"}
            mode: 检索模式，"dense" 使用稠密向量，"sparse" 使用稀疏向量

        Returns:
            检索结果列表: [{"memory": MemoryAtom, "score": float}, ...]
        """
        try:
            # 构建过滤条件 (支持 Dict 或 qdrant Filter 对象)
            if isinstance(filters, Filter):
                filter_obj = filters
            else:
                filter_obj = self._build_filter(filters) if filters else None

            if mode == "sparse":
                # BM25 检索 - 使用 Qdrant 原生 BM25 Document 查询
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(text=query_text, model="qdrant/bm25"),
                    using="sparse_text",
                    query_filter=filter_obj,
                    limit=top_k,
                    with_payload=True,
                )
                search_result = search_result.points
                logger.debug(f"✓ BM25 检索到 {len(search_result)} 条记忆")
            else:
                # 稠密向量检索 - 使用 query_points API
                query_vector = self.embedding_service.encode(dense_texts=query_text)
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="dense_text",  # 指定使用稠密向量配置
                    query_filter=filter_obj,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
                search_result = search_result.points  # 提取 points 列表
                logger.debug(f"✓ 稠密检索到 {len(search_result)} 条记忆")
                if len(search_result) == 0:
                    logger.warning("稠密检索返回 0 条结果。")

            # 解析结果
            results = []
            for hit in search_result:
                memory = self._payload_to_memory(hit.payload)
                results.append({
                    "memory": memory,
                    "score": hit.score,
                    "id": hit.id,
                })

            return results

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during search_memories: {e}")
            from hivememory.patchouli.mtp.exceptions import StorageOfflineError
            raise StorageOfflineError("Memory storage is unreachable.") from e
        except Exception as e:
            logger.error(f"Storage error during search_memories: {e}", exc_info=True)
            from hivememory.patchouli.mtp.exceptions import StorageReadError
            raise StorageReadError("Memory storage encountered an error during search.") from e

    def delete_memory(self, memory_id: UUID) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆UUID

        Returns:
            是否成功删除
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[str(memory_id)],
            )
            logger.debug(f"✓ 成功删除记忆: {memory_id}")
            return True

        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def update_access_info(self, memory_id: UUID) -> None:
        """
        更新记忆的访问信息(访问计数、最后访问时间)

        Args:
            memory_id: 记忆UUID
        """
        from datetime import datetime

        try:
            # 获取当前记忆
            memory = self.get_memory(memory_id)
            if not memory:
                return

            # 更新访问信息
            memory.meta.access_count += 1
            memory.meta.last_accessed_at = datetime.now()

            # 重新存储
            self.upsert_memory(memory)

        except Exception as e:
            logger.error(f"更新访问信息失败: {e}")

    def count_memories(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        统计记忆数量

        Args:
            filters: 过滤条件

        Returns:
            记忆数量
        """
        try:
            filter_obj = self._build_filter(filters) if filters else None
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_obj,
            )
            return result.count

        except Exception as e:
            logger.error(f"统计记忆数量失败: {e}")
            return 0

    def get_all_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[MemoryAtom]:
        """
        获取所有记忆（不分相似度排序）

        使用 Qdrant scroll API 获取所有满足条件的记忆，不进行向量检索。

        Args:
            filters: 过滤条件，如 {"meta.user_id": "123"}
            limit: 最多返回多少条（默认100）

        Returns:
            MemoryAtom 列表
        """
        try:
            filter_obj = self._build_filter(filters) if filters else None

            # 使用 scroll API 获取所有点
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=False,  # 不需要向量
            )

            # 解析结果
            memories = []
            for point in scroll_result[0]:  # scroll_result = (points, next_page_offset)
                memory = self._payload_to_memory(point.payload)
                memories.append(memory)

            logger.debug(f"✓ 获取到 {len(memories)} 条记忆")
            return memories

        except Exception as e:
            logger.error(f"获取所有记忆失败: {e}")
            return []

    def get_memories_by_vitality_range(
        self,
        min_vitality: float = 0.0,
        max_vitality: float = 1.0,
        limit: int = 100
    ) -> List[MemoryAtom]:
        """
        获取指定生命力范围的记忆

        用于垃圾回收器扫描低生命力记忆。

        Args:
            min_vitality: 最小生命力 (0-1)
            max_vitality: 最大生命力 (0-1)
            limit: 最大返回数量

        Returns:
            MemoryAtom 列表
        """
        try:
            # 构建生命力范围过滤条件
            filters = {
                "meta.vitality_score": {"gte": min_vitality, "lte": max_vitality}
            }

            filter_obj = self._build_filter(filters)

            # 使用 scroll API 获取记忆
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # 解析结果
            memories = []
            for point in scroll_result[0]:
                memory = self._payload_to_memory(point.payload)
                memories.append(memory)

            logger.debug(f"✓ 获取到 {len(memories)} 条记忆 (vitality: {min_vitality}-{max_vitality})")
            return memories

        except Exception as e:
            logger.error(f"按生命力范围获取记忆失败: {e}")
            return []

    def batch_delete_memories(self, memory_ids: List[UUID]) -> int:
        """
        批量删除记忆

        用于垃圾回收器批量归档后删除。

        Args:
            memory_ids: 记忆ID列表

        Returns:
            成功删除的数量
        """
        if not memory_ids:
            return 0

        try:
            # 转换为字符串ID列表
            str_ids = [str(mid) for mid in memory_ids]

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=str_ids,
            )

            logger.info(f"✓ 批量删除 {len(memory_ids)} 条记忆")
            return len(memory_ids)

        except Exception as e:
            logger.error(f"批量删除记忆失败: {e}")
            return 0

    # ========== 内部辅助方法 ==========

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        构建 Qdrant 过滤条件

        Args:
            filters: 字典格式的过滤条件，如 {"meta.user_id": "123"}

        Returns:
            Qdrant Filter 对象
        """
        must_conditions = []

        for key, value in filters.items():
            # Qdrant payload 字段直接使用 key，不需要 "payload." 前缀
            # 例如: "meta.user_id" 直接对应 payload 中的 meta.user_id
            field_path = key

            if isinstance(value, (str, int, bool)):
                must_conditions.append(
                    FieldCondition(key=field_path, match=MatchValue(value=value))
                )
            elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                # 范围查询 (如 confidence_score >= 0.8)
                must_conditions.append(
                    FieldCondition(
                        key=field_path,
                        range=Range(
                            gte=value.get("gte"),
                            lte=value.get("lte"),
                        ),
                    )
                )

        return Filter(must=must_conditions) if must_conditions else None

    def _payload_to_memory(self, payload: Dict[str, Any]) -> MemoryAtom:
        """
        将 Qdrant Payload 转换回 MemoryAtom 对象

        Args:
            payload: Qdrant 存储的 payload

        Returns:
            MemoryAtom 对象
        """
        from hivememory.core.models import MetaData, PayloadLayer, RelationLayer

        return MemoryAtom(
            id=UUID(payload["id"]),
            meta=MetaData(**payload["meta"]),
            index=IndexLayer(**payload["index"]),
            payload=PayloadLayer(**payload["payload"]),
            relations=RelationLayer(**payload.get("relations", {})),
        )
