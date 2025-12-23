"""
Qdrant 向量存储层封装

提供:
- 集合管理(创建、删除)
- 记忆原子的 CRUD 操作
- 混合检索(向量 + 元数据过滤)
- 批量操作
"""

from typing import List, Optional, Dict, Any
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
    SearchRequest,
)
from sentence_transformers import SentenceTransformer

from hivememory.core.models import MemoryAtom, IndexLayer
from hivememory.core.config import QdrantConfig, EmbeddingConfig, get_config

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
        qdrant_config: Optional[QdrantConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        """
        初始化存储管理器

        Args:
            qdrant_config: Qdrant 配置
            embedding_config: Embedding 配置
        """
        # 加载配置
        if qdrant_config is None or embedding_config is None:
            global_config = get_config()
            qdrant_config = qdrant_config or global_config.qdrant
            embedding_config = embedding_config or global_config.embedding

        self.qdrant_config = qdrant_config
        self.embedding_config = embedding_config

        # 初始化 Qdrant 客户端
        # 本地部署时 api_key 为 None，不传递给客户端
        client_kwargs = {
            "host": qdrant_config.host,
            "port": qdrant_config.port,
            "timeout": 60,
        }

        # 只有在 api_key 有值时才添加
        if qdrant_config.api_key and qdrant_config.api_key.strip():
            client_kwargs["api_key"] = qdrant_config.api_key

        self.client = QdrantClient(**client_kwargs)

        # 初始化 Embedding 模型
        logger.info(f"加载 Embedding 模型: {embedding_config.model_name}")
        self.embedding_model = SentenceTransformer(
            embedding_config.model_name,
            device=embedding_config.device
        )

        self.collection_name = qdrant_config.collection_name
        self.vector_dimension = qdrant_config.vector_dimension

    def create_collection(self, recreate: bool = False) -> None:
        """
        创建向量集合

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
                    logger.info(f"集合已存在: {self.collection_name}")
                    return

            # 创建新集合
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=getattr(Distance, self.qdrant_config.distance_metric.upper()),
                ),
                on_disk_payload=self.qdrant_config.on_disk_payload,
            )

            logger.info(f"✓ 成功创建集合: {self.collection_name}")

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        生成文本的 Embedding 向量

        Args:
            text: 待编码文本

        Returns:
            向量列表
        """
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=self.embedding_config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def upsert_memory(self, memory: MemoryAtom) -> None:
        """
        插入或更新记忆原子

        Args:
            memory: 记忆原子对象

        Raises:
            Exception: 操作失败时抛出
        """
        try:
            # 1. 生成 Embedding 向量
            if memory.index.embedding is None:
                embedding_text = memory.index.get_embedding_text()
                embedding = self.generate_embedding(embedding_text)
                memory.index.embedding = embedding
            else:
                embedding = memory.index.embedding

            # 2. 构建 Qdrant Point
            point = PointStruct(
                id=str(memory.id),
                vector=embedding,
                payload=memory.to_qdrant_payload(),
            )

            # 3. 插入数据库
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.debug(f"✓ 成功存储记忆: {memory.id} - {memory.index.title}")

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

        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return None

    def search_memories(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义检索记忆

        Args:
            query_text: 查询文本
            top_k: 返回Top K结果
            score_threshold: 最低相似度阈值
            filters: 元数据过滤条件, 如 {"memory_type": "CODE_SNIPPET", "user_id": "123"}

        Returns:
            检索结果列表: [{"memory": MemoryAtom, "score": float}, ...]
        """
        try:
            # 1. 生成查询向量
            query_vector = self.generate_embedding(query_text)

            # 2. 构建过滤条件
            filter_obj = self._build_filter(filters) if filters else None

            # 3. 执行检索 (使用新版 Qdrant API)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=filter_obj,
                score_threshold=score_threshold,
                with_payload=True,
            ).points

            # 4. 解析结果
            results = []
            for hit in search_result:
                memory = self._payload_to_memory(hit.payload)
                results.append({
                    "memory": memory,
                    "score": hit.score,
                    "id": hit.id,
                })

            logger.debug(f"✓ 检索到 {len(results)} 条记忆")
            return results

        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []

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

    # ========== 内部辅助方法 ==========

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        构建 Qdrant 过滤条件

        Args:
            filters: 字典格式的过滤条件

        Returns:
            Qdrant Filter 对象
        """
        must_conditions = []

        for key, value in filters.items():
            # 处理嵌套字段 (如 "meta.user_id")
            field_path = f"payload.{key}" if "." not in key else f"payload.{key}"

            if isinstance(value, (str, int, bool)):
                must_conditions.append(
                    FieldCondition(key=field_path, match=MatchValue(value=value))
                )
            elif isinstance(value, dict) and "gte" in value or "lte" in value:
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


# 便捷函数: 获取全局单例
_global_store: Optional[QdrantMemoryStore] = None


def get_memory_store() -> QdrantMemoryStore:
    """获取全局 QdrantMemoryStore 实例 (单例模式)"""
    global _global_store
    if _global_store is None:
        _global_store = QdrantMemoryStore()
    return _global_store
