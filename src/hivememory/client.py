"""
HiveMemory 统一客户端入口

这是用户（开发者）与 HiveMemory 系统交互的主要入口。

作者: HiveMemory Team
版本: 2.0
"""

import logging
from typing import List, Optional, Dict, Any

from hivememory.core.models import MemoryAtom
from hivememory.engines.generation.models import ConversationMessage
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.patchouli.config import HiveMemoryConfig, load_app_config
from hivememory.patchouli.system import PatchouliSystem
from hivememory.patchouli.eye import TheEye
from hivememory.patchouli.retrieval_familiar import RetrievalFamiliar
from hivememory.patchouli.librarian_core import LibrarianCore

logger = logging.getLogger(__name__)


class HiveMemoryClient:
    """
    HiveMemory 统一客户端

    这是对外提供的唯一入口，封装了完整的帕秋莉体系。

    架构:
        - TheEye (真理之眼): 意图识别、查询重写
        - RetrievalFamiliar (检索使魔): 混合检索、上下文渲染
        - LibrarianCore (馆长本体): 感知缓冲、记忆生成、生命周期管理

    使用示例:
        >>> from hivememory.client import HiveMemoryClient
        >>>
        >>> # 使用默认配置
        >>> client = HiveMemoryClient()
        >>>
        >>> # 处理查询
        >>> result = client.process_query(
        ...     query="我之前设置的 API Key 是什么？",
        ...     context=[],
        ...     user_id="user123"
        ... )
        >>>
        >>> # 添加对话
        >>> client.add_conversation(
        ...     role="user",
        ...     content="帮我写贪吃蛇游戏",
        ...     user_id="user123"
        ... )
    """

    def __init__(
        self,
        config: Optional[HiveMemoryConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        初始化 HiveMemory 客户端

        Args:
            config: 配置对象（可选）
            config_path: 配置文件路径（可选）

        Examples:
            >>> # 使用默认配置
            >>> client = HiveMemoryClient()
            >>>
            >>> # 使用自定义配置文件
            >>> client = HiveMemoryClient(config_path="path/to/config.yaml")
            >>>
            >>> # 使用配置对象
            >>> from hivememory.patchouli.config import HiveMemoryConfig
            >>> config = HiveMemoryConfig()
            >>> client = HiveMemoryClient(config=config)
        """
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_app_config(config_path)
        else:
            self.config = load_app_config()

        # 初始化系统
        self.system = PatchouliSystem(config=self.config)

        logger.info("HiveMemoryClient 初始化完成")

    # ========== 系统属性访问 ==========

    @property
    def eye(self) -> TheEye:
        """访问真理之眼"""
        return self.system.eye

    @property
    def retrieval_familiar(self) -> RetrievalFamiliar:
        """访问检索使魔"""
        return self.system.retrieval_familiar

    @property
    def librarian_core(self) -> LibrarianCore:
        """访问馆长本体"""
        return self.system.librarian_core

    @property
    def storage(self) -> QdrantMemoryStore:
        """访问存储层"""
        return self.system.storage

    # ========== 主要 API ==========

    def process_query(
        self,
        query: str,
        context: Optional[List[ConversationMessage]] = None,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        """
        处理用户查询（Hot Path）

        这是主要入口，整合了意图判断、查询重写、记忆检索。

        Args:
            query: 用户原始查询
            context: 对话上下文
            user_id: 用户 ID
            agent_id: Agent ID

        Returns:
            Dict: 包含 intent, rewritten, memory 等字段
        """
        return self.system.process_user_query(query, context or [], user_id, agent_id)

    def add_conversation(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
    ) -> None:
        """
        添加对话到感知层（Cold Path）

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID（可选）
        """
        self.system.perceive(role, content, user_id, agent_id, session_id)

    def retrieve_memory(
        self,
        query: str,
        user_id: str,
        context: Optional[List[ConversationMessage]] = None,
        **kwargs
    ) -> str:
        """
        检索记忆（Hot Path 快捷入口）

        Args:
            query: 查询文本
            user_id: 用户 ID
            context: 对话上下文（可选）
            **kwargs: 其他检索参数

        Returns:
            str: 渲染后的记忆上下文
        """
        return self.system.retrieve(query, user_id, context, **kwargs)

    def flush_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        """手动触发感知层 Flush"""
        self.system.flush_buffer(user_id, agent_id, session_id)

    def get_buffer_info(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """获取 Buffer 信息"""
        return self.system.get_buffer_info(user_id, agent_id, session_id)

    def add_flush_observer(self, observer) -> None:
        """添加 Flush 事件观察者"""
        self.system.add_flush_observer(observer)

    # ========== 存储层便捷方法 ==========

    def get_all_memories(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[MemoryAtom]:
        """
        获取所有记忆

        Args:
            filters: 过滤条件
            limit: 最大返回数量

        Returns:
            List[MemoryAtom]: 记忆列表
        """
        return self.storage.get_all_memories(filters=filters, limit=limit)

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryAtom]:
        """
        根据 ID 获取记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            Optional[MemoryAtom]: 记忆对象，不存在时返回 None
        """
        return self.storage.get_memory_by_id(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            bool: 是否成功删除
        """
        return self.storage.delete_memory(memory_id)


__all__ = [
    "HiveMemoryClient",
]
