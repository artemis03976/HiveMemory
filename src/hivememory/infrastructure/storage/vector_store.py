"""
Qdrant 向量存储层封装

提供:
- 集合管理(创建、删除)
- 记忆原子的 CRUD 操作
- 混合检索(向量 + 元数据过滤)
- 批量操作
"""

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    AgentProfile,
    MemoryAtom,
    IdentityScope,
    MemoryType,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.mtp.exceptions import (
    AliasNotFoundError,
    InvalidArgumentError,
    MemoryTypeMismatchError,
)
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget
from hivememory.engines.retrieval.memory_codec import MemoryDecodeError, decode_memory_payload
from hivememory.engines.retrieval.policy import memory_belongs_to_workspace
from hivememory.infrastructure.embedding import get_bge_m3_service
from hivememory.infrastructure.storage.qdrant_client import (
    create_async_qdrant_client,
    wait_for_qdrant_ready,
)
from hivememory.system.config import EmbeddingConfig, QdrantConfig

logger = logging.getLogger(__name__)

_compiler = MemoryCompiler()


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

        logger.info(f"加载 BGE-M3 Embedding 服务")

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
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

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
                sparse_vectors_config={
                    "sparse_text": SparseVectorParams(modifier=Modifier.IDF)
                },
                on_disk_payload=self.qdrant_config.on_disk_payload,
            )

            logger.info(f"✓ 成功创建集合: {self.collection_name} (Dense + Sparse)")

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    async def upsert_memory(
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
                dense_text = _compiler.compile(memory, MemoryCompileTarget.DENSE_EMBEDDING).text
                sparse_context = _compiler.compile(memory, MemoryCompileTarget.SPARSE_EMBEDDING).text
                vectors = self.embedding_service.encode(
                    dense_texts=dense_text,
                    sparse_texts=sparse_context
                )

                # 构建 Qdrant Point - dense 向量 + BM25 文本
                sparse_text = vectors["sparse_text"]
                point = PointStruct(
                    id=self._point_id(
                        WorkspaceMemoryKey(
                            workspace_identity=memory.workspace_identity,
                            memory_id=memory.id,
                        )
                    ),
                    vector={
                        "dense_text": vectors["dense"],
                        "sparse_text": Document(text=sparse_text, model="qdrant/bm25"),
                    },
                    payload=memory.to_qdrant_payload(),
                )
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
                await self._remove_legacy_point_after_upsert(memory)
                logger.debug(f"✓ 成功存储记忆 (Dense+Sparse): {memory.id} - {memory.index.title}")
            else:
                # 仅使用稠密向量
                embedding_text = _compiler.compile(memory, MemoryCompileTarget.DENSE_EMBEDDING).text
                embedding = self.embedding_service.encode(dense_texts=embedding_text)

                # 构建 Qdrant Point - 使用命名向量格式以保持一致性
                point = PointStruct(
                    id=self._point_id(
                        WorkspaceMemoryKey(
                            workspace_identity=memory.workspace_identity,
                            memory_id=memory.id,
                        )
                    ),
                    vector={
                        "dense_text": embedding,
                    },
                    payload=memory.to_qdrant_payload(),
                )
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                )
                await self._remove_legacy_point_after_upsert(memory)
                logger.debug(f"✓ 成功存储记忆 (Dense): {memory.id} - {memory.index.title}")

        except Exception as e:
            logger.error(f"存储记忆失败: {e}")
            raise

    async def get_memory(self, key: WorkspaceMemoryKey) -> Optional[MemoryAtom]:
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError

        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[self._point_id(key)],
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                # 仅 compatibility-read 检查旧版全局 UUID 点；归属仍须重验。
                points = await self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[str(key.memory_id)],
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    return None

            memory = self._payload_to_memory(points[0].payload)
            if not memory_belongs_to_workspace(memory, key.workspace_identity):
                return None
            return memory

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during get_memory: {e}")
            raise StorageOfflineError(cause=e) from e
        except (UnexpectedResponse, ResponseHandlingException) as e:
            logger.error(f"Storage error during get_memory: {e}")
            raise StorageReadError(cause=e) from e
        except Exception as e:
            logger.error(f"Unexpected storage error in get_memory: {e}", exc_info=True)
            raise StorageReadError(cause=e) from e

    async def get_memory_by_alias(
        self,
        alias: str,
        *,
        query_filter: Filter,
        workspace_identity: WorkspaceIdentity,
    ) -> Optional[MemoryAtom]:
        """
        根据别名精确匹配检索记忆 (L2 Cold Lookup, MTP Section 2.3.2)

        使用 scroll API + FieldCondition 精确匹配 index.alias 字段。

        Args:
            alias: 语义化别名 (e.g. "code_quicksort_impl")
            workspace_identity: 已验证的 Workspace ownership 边界

        Returns:
            MemoryAtom 对象，未找到返回 None
        """
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        from hivememory.core.mtp.exceptions import StorageOfflineError, StorageReadError

        try:
            alias_filter = FieldCondition(
                key="index.alias",
                match=MatchValue(value=alias),
            )
            filter_obj = Filter(must=[query_filter, alias_filter])

            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )

            points = scroll_result[0]
            if not points:
                return None

            memory = self._payload_to_memory(points[0].payload)
            if not memory_belongs_to_workspace(memory, workspace_identity):
                return None
            return memory

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during get_memory_by_alias (alias={alias}): {e}")
            raise StorageOfflineError(cause=e) from e
        except (UnexpectedResponse, ResponseHandlingException) as e:
            logger.error(f"Storage error during get_memory_by_alias (alias={alias}): {e}")
            raise StorageReadError(cause=e) from e
        except Exception as e:
            logger.error(f"Unexpected storage error in get_memory_by_alias (alias={alias}): {e}", exc_info=True)
            raise StorageReadError(cause=e) from e

    async def get_agent_profile(
        self,
        scope: IdentityScope,
        agent_alias: str | None,
    ) -> AgentProfile:
        """在显式 Memory scope 内读取 Agent profile。"""
        normalized_alias = agent_alias.strip() if agent_alias else ""
        if not normalized_alias or normalized_alias in ("default", "omni_doll"):
            return OMNI_DOLL_PROFILE

        from hivememory.engines.retrieval.filter_adapter import QdrantFilterConverter
        from hivememory.engines.retrieval.models import QueryFilters

        atom = await self.get_memory_by_alias(
            normalized_alias,
            query_filter=QdrantFilterConverter().convert(QueryFilters(), scope),
            workspace_identity=scope.workspace_identity,
        )
        if atom is None:
            raise AliasNotFoundError(
                message_key="mtp.call.profile_not_found",
                params={"agent_alias": normalized_alias},
            )
        if atom.index.memory_type != MemoryType.AGENT_PROFILE:
            raise MemoryTypeMismatchError(
                message_key="mtp.call.profile_type_mismatch",
                params={"agent_alias": normalized_alias},
            )
        profile = AgentProfile.from_atom(atom)
        if profile is None:
            raise InvalidArgumentError(
                message_key="mtp.call.profile_invalid",
                params={"agent_alias": normalized_alias},
            )
        return profile

    async def search_memories(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Union[Dict[str, Any], Filter]] = None,
        mode: str = "dense",
        *,
        workspace_identity: WorkspaceIdentity,
    ) -> List[Dict[str, Any]]:
        """
        语义检索记忆 (支持稠密和稀疏向量检索)

        Args:
            query_text: 查询文本
            top_k: 返回Top K结果
            score_threshold: 最低相似度阈值
            filters: 已构造的元数据过滤条件，必须包含 Workspace ownership 边界
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
                search_result = await self.client.query_points(
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
                search_result = await self.client.query_points(
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
                try:
                    memory = self._payload_to_memory(hit.payload)
                except MemoryDecodeError as exc:
                    logger.warning(
                        "拒绝无法安全解码的 Memory 搜索结果: point_id=%s, error=%s",
                        hit.id,
                        exc,
                    )
                    continue
                if not memory_belongs_to_workspace(memory, workspace_identity):
                    logger.warning(
                        "拒绝 Workspace 归属不一致的 Memory 搜索结果: point_id=%s",
                        hit.id,
                    )
                    continue
                results.append({
                    "memory": memory,
                    "score": hit.score,
                    "id": hit.id,
                })

            return results

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Storage offline during search_memories: {e}")
            from hivememory.core.mtp.exceptions import StorageOfflineError
            raise StorageOfflineError(cause=e) from e
        except Exception as e:
            logger.error(f"Storage error during search_memories: {e}", exc_info=True)
            from hivememory.core.mtp.exceptions import StorageReadError
            raise StorageReadError(cause=e) from e

    async def delete_memory(self, key: WorkspaceMemoryKey) -> bool:
        try:
            point_ids = [self._point_id(key)]
            if await self._legacy_point_belongs_to(key):
                point_ids.append(str(key.memory_id))
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            logger.debug(f"✓ 成功删除记忆: {key.memory_id}")
            return True

        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    async def count_memories(
        self,
        filters: Optional[Union[Dict[str, Any], Filter]] = None,
    ) -> int:
        try:
            filter_obj = filters if isinstance(filters, Filter) else (
                self._build_filter(filters) if filters else None
            )
            result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_obj,
            )
            return result.count

        except Exception as e:
            logger.error(f"统计记忆数量失败: {e}")
            return 0

    async def get_all_memories(
        self,
        *,
        filters: Union[Dict[str, Any], Filter],
        workspace_identity: WorkspaceIdentity,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        """
        获取所有记忆（不分相似度排序）

        使用 Qdrant scroll API 获取所有满足条件的记忆，不进行向量检索。

        Args:
            filters: 过滤条件，如 {"meta.workspace_identity.workspace_id": "main_workspace"}
            limit: 最多返回多少条（默认100）

        Returns:
            MemoryAtom 列表
        """
        try:
            filter_obj = filters if isinstance(filters, Filter) else self._build_filter(filters)

            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # 解析结果
            memories = []
            for point in scroll_result[0]:
                try:
                    memory = self._payload_to_memory(point.payload)
                except MemoryDecodeError as exc:
                    logger.warning(
                        "拒绝无法安全解码的 Memory scroll 结果: point_id=%s, error=%s",
                        point.id,
                        exc,
                    )
                    continue
                if not memory_belongs_to_workspace(memory, workspace_identity):
                    logger.warning(
                        "拒绝 Workspace 归属不一致的 Memory scroll 结果: point_id=%s",
                        point.id,
                    )
                    continue
                memories.append(memory)

            logger.debug(f"✓ 获取到 {len(memories)} 条记忆")
            return memories

        except Exception as e:
            logger.error(f"获取所有记忆失败: {e}")
            return []

    async def get_all_memories_for_maintenance(
        self,
        *,
        limit: int = 10000,
    ) -> list[MemoryAtom]:
        """进程级维护遍历；legacy owner 仅由版本解码器归一化。"""
        try:
            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            memories: list[MemoryAtom] = []
            for point in scroll_result[0]:
                try:
                    memories.append(self._payload_to_memory(point.payload))
                except MemoryDecodeError as exc:
                    logger.warning(
                        "维护遍历拒绝歧义 Memory: point_id=%s, error=%s",
                        point.id,
                        exc,
                    )
            return memories
        except Exception as exc:
            logger.error("维护遍历 Memory 失败: %s", exc, exc_info=True)
            return []

    async def get_memories_by_vitality_range(
        self,
        min_vitality: float = 0.0,
        max_vitality: float = 100.0,
        limit: int = 100
    ) -> List[MemoryAtom]:
        """
        获取指定生命力范围的记忆

        用于垃圾回收器扫描低生命力记忆。

        Args:
            min_vitality: 最小生命力 (0-100)
            max_vitality: 最大生命力 (0-100)
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

            scroll_result = await self.client.scroll(
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

    async def batch_delete_memories(self, keys: List[WorkspaceMemoryKey]) -> int:
        if not keys:
            return 0

        try:
            point_ids = [self._point_id(key) for key in keys]
            for key in keys:
                if await self._legacy_point_belongs_to(key):
                    point_ids.append(str(key.memory_id))

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )

            logger.info(f"✓ 批量删除 {len(keys)} 条记忆")
            return len(keys)

        except Exception as e:
            logger.error(f"批量删除记忆失败: {e}")
            return 0

    # ========== 内部辅助方法 ==========

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        构建 Qdrant 过滤条件

        Args:
            filters: 字典格式的过滤条件，如 {"meta.workspace_identity.workspace_id": "main_workspace"}

        Returns:
            Qdrant Filter 对象
        """
        must_conditions = []

        for key, value in filters.items():
            # Qdrant payload 字段直接使用 key，不需要 "payload." 前缀
            # 例如: "meta.workspace_identity.workspace_id" 直接对应嵌套 payload 字段
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
        return decode_memory_payload(payload)

    async def _remove_legacy_point_after_upsert(self, memory: MemoryAtom) -> None:
        """仅回收同一 Workspace 的旧 UUID 点，避免跨域 ID 碰撞误删。"""
        key = WorkspaceMemoryKey(
            workspace_identity=memory.workspace_identity,
            memory_id=memory.id,
        )
        if not await self._legacy_point_belongs_to(key):
            return
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[str(memory.id)],
            )
        except Exception as exc:
            logger.warning(
                "Memory v2 已写入，但旧版 UUID 点清理失败: memory_id=%s, error=%s",
                memory.id,
                exc,
            )

    async def _legacy_point_belongs_to(self, key: WorkspaceMemoryKey) -> bool:
        """确认旧全局 UUID 点归属当前复合键，未确认时宁可保留。"""
        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(key.memory_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return False
            memory = self._payload_to_memory(points[0].payload)
        except MemoryDecodeError as exc:
            logger.warning(
                "拒绝清理无法安全解码的 legacy Memory 点: memory_id=%s, error=%s",
                key.memory_id,
                exc,
            )
            return False
        except Exception as exc:
            logger.warning(
                "legacy Memory 点归属核验失败，保留原点: memory_id=%s, error=%s",
                key.memory_id,
                exc,
            )
            return False
        return memory_belongs_to_workspace(memory, key.workspace_identity)

    @staticmethod
    def _point_id(key: WorkspaceMemoryKey) -> str:
        """将复合资源键稳定映射为 Qdrant 支持的 UUID point id。"""
        workspace = key.workspace_identity
        canonical = (
            f"hivememory-memory:{workspace.owner_user_id}:"
            f"{workspace.workspace_id}:{key.memory_id}"
        )
        return str(uuid5(NAMESPACE_URL, canonical))
