"""
Mock 存储服务

用于集成测试中替代真实的 QdrantMemoryStore。
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4
from collections import defaultdict

from hivememory.core.models import MemoryAtom, QueryFilters


class MockMemoryStore:
    """
    Mock 内存存储，用于测试组件协作。

    不连接真实数据库，使用内存字典存储。
    """

    def __init__(self):
        self._memories: Dict[str, MemoryAtom] = {}
        self._user_memories: defaultdict = defaultdict(list)
        self._call_count = defaultdict(int)

    def upsert_memory(
        self,
        memory: MemoryAtom,
        use_sparse: bool = False,
    ) -> str:
        """插入或更新���忆"""
        self._call_count["upsert"] += 1
        self._memories[memory.id] = memory
        self._user_memories[memory.meta.user_id].append(memory.id)
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[MemoryAtom]:
        """获取单条记忆"""
        self._call_count["get"] += 1
        return self._memories.get(memory_id)

    def get_memories_by_filter(
        self,
        filters: QueryFilters,
        limit: int = 10,
    ) -> List[MemoryAtom]:
        """按过滤条件获取记忆"""
        self._call_count["filter"] += 1
        results = []

        for memory in self._memories.values():
            if filters.memory_type and memory.index.memory_type != filters.memory_type:
                continue
            if filters.user_id and memory.meta.user_id != filters.user_id:
                continue
            if filters.tags:
                if not any(tag in memory.index.tags for tag in filters.tags):
                    continue
            results.append(memory)
            if len(results) >= limit:
                break

        return results

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[QueryFilters] = None,
    ) -> List[tuple[MemoryAtom, float]]:
        """向量相似度搜索（Mock实现）"""
        self._call_count["search"] += 1
        # 简单返回前N条，带模拟分数
        results = []
        for memory in list(self._memories.values())[:limit]:
            # 模拟分数：0.5-0.95之间
            score = 0.5 + (len(memory.id) % 45) / 100.0
            results.append((memory, score))
        return results

    def count_memories(self, user_id: Optional[str] = None) -> int:
        """统计记忆数量"""
        self._call_count["count"] += 1
        if user_id:
            return len(self._user_memories.get(user_id, []))
        return len(self._memories)

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        self._call_count["delete"] += 1
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            # 从用户索引中移除
            if memory_id in self._user_memories[memory.meta.user_id]:
                self._user_memories[memory.meta.user_id].remove(memory_id)
            del self._memories[memory_id]
            return True
        return False

    def clear_all(self):
        """清空所有记忆"""
        self._memories.clear()
        self._user_memories.clear()

    def get_call_stats(self) -> Dict[str, int]:
        """获取调用统计（用于验证）"""
        return dict(self._call_count)

    def reset_stats(self):
        """重置统计"""
        self._call_count.clear()
