"""
Patchouli - Librarian Agent (图书管理员智能体)

职责:
    1. 记忆感知
    2. 记忆生成与写入记忆库
    3. 记忆检索与上下文注入
    4. 生命周期管理（Stage 3 - 待实现）

架构:
    - SimplePerceptionLayer: 简单缓冲策略（三重触发）
    - SemanticFlowPerceptionLayer: 语义流策略（统一语义流）

参考: PROJECT.md 4.0 - 4.2 节

作者: HiveMemory Team
版本: 0.4.0 (感知层重构)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING

from hivememory.core.models import MemoryAtom, FlushReason
from hivememory.generation.models import ConversationMessage
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.generation.orchestrator import MemoryOrchestrator
from hivememory.perception.interfaces import BasePerceptionLayer

if TYPE_CHECKING:
    from hivememory.core.config import MemoryPerceptionConfig

logger = logging.getLogger(__name__)


@dataclass
class FlushEvent:
    """
    Flush 事件数据，用于外部观察

    Attributes:
        messages: 被 Flush 的消息列表
        reason: Flush 触发原因
        memories: 生成的记忆原子列表（可能为空）
        timestamp: 事件发生时间戳
    """
    messages: List[ConversationMessage]
    reason: FlushReason
    memories: List[MemoryAtom]
    timestamp: float = field(default_factory=time.time)


# 观察者类型别名
FlushObserver = Callable[[FlushEvent], None]


class PatchouliAgent:
    """
    Patchouli - HiveMemory 的图书管理员

    性格特征 (Persona):
        - 极度挑剔: 只记录有价值的信息,拒绝噪音
        - 结构化强迫症: 致力于将混乱的对话转化为规范的JSON
        - 客观中立: 以第三人称视角记录事实

    使用示例:
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>>
        >>> # 使用默认配置（语义流感知层）
        >>> patchouli = PatchouliAgent(storage=storage)
        >>>
        >>> # 使用简单感知层
        >>> from hivememory.core.config import MemoryPerceptionConfig
        >>> config = MemoryPerceptionConfig(layer_type="simple")
        >>> patchouli = PatchouliAgent(storage=storage, perception_config=config)
        >>>
        >>> # 添加消息
        >>> patchouli.add_message("user", "帮我写贪吃蛇游戏", "user123", "agent1")
        >>> patchouli.add_message("assistant", "好的，我来实现...", "user123", "agent1")
    """

    def __init__(
        self,
        storage: Optional[QdrantMemoryStore] = None,
        perception_config: Optional["MemoryPerceptionConfig"] = None,
    ):
        """
        初始化 Patchouli Agent

        Args:
            storage: Qdrant 存储实例（可选，自动创建）
            perception_config: 感知层配置（可选，使用默认配置）

        Examples:
            >>> # 使用默认配置（语义流感知层）
            >>> patchouli = PatchouliAgent()
            >>>
            >>> # 使用简单感知层
            >>> from hivememory.core.config import MemoryPerceptionConfig
            >>> config = MemoryPerceptionConfig(layer_type="simple")
            >>> patchouli = PatchouliAgent(perception_config=config)
            >>>
            >>> # 使用自定义语义阈值
            >>> config = MemoryPerceptionConfig()
            >>> config.semantic_flow.semantic_threshold = 0.8
            >>> patchouli = PatchouliAgent(perception_config=config)
        """
        self.storage = storage or QdrantMemoryStore()
        self.perception_config = perception_config
        self.perception_layer: Optional[BasePerceptionLayer] = None

        # 初始化感知层
        self._init_perception_layer()

        # 初始化记忆生成编排器（Stage 1: 记忆生成核心逻辑）
        self.orchestrator = MemoryOrchestrator(storage=self.storage)

        # Flush 事件观察者列表
        self._flush_observers: List[FlushObserver] = []

        # 记录实际使用的感知层类型
        layer_type = getattr(self.perception_config, "layer_type", "semantic_flow") if self.perception_config else "semantic_flow"
        logger.info(f"Patchouli Agent 初始化完成 (perception_layer_type={layer_type})")

    def _init_perception_layer(self) -> None:
        """
        初始化感知层

        使用统一的工厂函数创建感知层，根据配置的 layer_type 自动选择：
        - "semantic_flow": SemanticFlowPerceptionLayer（语义流策略）
        - "simple": SimplePerceptionLayer（简单触发策略）

        工厂函数会从 config.yaml 加载默认配置，或使用传入的自定义配置。
        """
        try:
            from hivememory.perception import create_default_perception_layer

            # 使用工厂函数创建感知层
            # 如果未提供配置，工厂函数会从 config.yaml 加载默认配置
            self.perception_layer = create_default_perception_layer(
                on_flush_callback=self._on_perception_flush,
                config=self.perception_config,
            )

            if self.perception_layer:
                layer_name = type(self.perception_layer).__name__
                logger.debug(f"{layer_name} 已初始化")
            else:
                logger.warning("感知层未启用 (config.enable=False)")

        except ImportError as e:
            logger.error(f"无法导入感知层: {e}")
            raise

    # ========== 感知层 API ==========

    def add_message(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
    ) -> None:
        """
        添加消息到感知层

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID（可选）

        Examples:
            >>> patchouli.add_message("user", "你好", "user123", "chatbot")
            >>> patchouli.add_message("assistant", "你好！有什么可以帮你的吗？", "user123", "chatbot")
        """
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.add_message(
            role=role,
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id or f"{user_id}_{agent_id}",
        )

        logger.debug(f"向感知层添加消息: {role} - {content[:50]}...")

    def flush_perception(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        """
        手动触发感知层 Flush

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Examples:
            >>> patchouli.flush_perception("user123", "chatbot", "session_456")
        """
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.flush_buffer(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    def _on_perception_flush(
        self,
        messages: List[ConversationMessage],
        reason: FlushReason,
    ) -> None:
        """
        感知层 Flush 回调（统一接口）
        将感知层生成的消息传递给编排器处理

        Args:
            messages: ConversationMessage 列表
            reason: Flush 原因
        """
        try:
            # 从消息中提取上下文
            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            # TODO: 获取正确的 user_id 和 agent_id
            first_msg = messages[0]
            user_id = getattr(first_msg, "user_id", "unknown")
            agent_id = "perception_layer"

            logger.info(f"Patchouli 开始处理 {len(messages)} 条消息...")

            # 调用编排器处理
            memories = self.orchestrator.process(
                messages=messages,
            )

            # 创建 Flush 事件并通知观察者
            event = FlushEvent(
                messages=messages,
                reason=reason,
                memories=memories or [],
            )
            self._notify_flush_observers(event)

            logger.info(f"Patchouli 处理完成")
            if memories:
                logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

    # ========== 观察者模式 API ==========

    def add_flush_observer(self, observer: FlushObserver) -> None:
        """
        添加 Flush 事件观察者

        观察者会在每次 Flush 事件发生后被调用，可用于：
        - 测试时捕获 Flush 事件和生成的记忆
        - 监控和日志记录
        - 与其他系统集成

        Args:
            observer: 回调函数，签名为 Callable[[FlushEvent], None]

        Examples:
            >>> def my_observer(event: FlushEvent):
            ...     print(f"Flush: {event.reason}, Memories: {len(event.memories)}")
            >>> patchouli.add_flush_observer(my_observer)
        """
        self._flush_observers.append(observer)
        logger.debug(f"添加 Flush 观察者，当前共 {len(self._flush_observers)} 个")

    def remove_flush_observer(self, observer: FlushObserver) -> None:
        """
        移除 Flush 事件观察者

        Args:
            observer: 要移除的回调函数

        Examples:
            >>> patchouli.remove_flush_observer(my_observer)
        """
        if observer in self._flush_observers:
            self._flush_observers.remove(observer)
            logger.debug(f"移除 Flush 观察者，当前共 {len(self._flush_observers)} 个")

    def _notify_flush_observers(self, event: FlushEvent) -> None:
        """
        通知所有观察者

        Args:
            event: Flush 事件数据
        """
        for observer in self._flush_observers:
            try:
                observer(event)
            except Exception as e:
                logger.warning(f"Flush 观察者执行失败: {e}")

    # ========== Buffer 管理 API ==========

    def get_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Buffer 实例（SimpleBuffer 或 SemanticBuffer）
        """
        if not self.perception_layer:
            return None

        return self.perception_layer.get_buffer(user_id, agent_id, session_id)

    def clear_buffer(self, user_id: str, agent_id: str, session_id: str) -> bool:
        """
        清理指定的 Buffer

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            bool: 是否成功清理

        Examples:
            >>> patchouli.clear_buffer("user123", "chatbot", "session_456")
            True
        """
        if not self.perception_layer:
            return False

        return self.perception_layer.clear_buffer(user_id, agent_id, session_id)

    def get_buffer_info(self, user_id: str, agent_id: str, session_id: str) -> Dict[str, Any]:
        """
        获取 Buffer 信息（统一接口）

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Dict: Buffer 信息字典

        Examples:
            >>> info = patchouli.get_buffer_info("user123", "chatbot", "session_456")
            >>> print(f"消息数量: {info.get('block_count', info.get('message_count', 0))}")
        """
        if not self.perception_layer:
            return {"exists": False, "mode": "none"}

        return self.perception_layer.get_buffer_info(user_id, agent_id, session_id)

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表

        Examples:
            >>> buffers = patchouli.list_active_buffers()
            >>> print(f"当前有 {len(buffers)} 个活跃 Buffer")
        """
        if not self.perception_layer:
            return []

        return self.perception_layer.list_active_buffers()


# 便捷函数
def create_patchouli_agent(
    storage: Optional[QdrantMemoryStore] = None,
    layer_type: str = "semantic_flow",
) -> PatchouliAgent:
    """
    创建 Patchouli Agent 实例

    Args:
        storage: Qdrant 存储实例（可选）
        layer_type: 感知层类型，"semantic_flow" 或 "simple"（默认 "semantic_flow"）

    Returns:
        PatchouliAgent: Agent 实例

    Examples:
        >>> # 使用语义流感知层（默认）
        >>> patchouli = create_patchouli_agent()
        >>>
        >>> # 使用简单感知层
        >>> patchouli = create_patchouli_agent(layer_type="simple")
    """
    from hivememory.core.config import MemoryPerceptionConfig

    config = MemoryPerceptionConfig(layer_type=layer_type)
    return PatchouliAgent(storage=storage, perception_config=config)


__all__ = [
    "PatchouliAgent",
    "FlushEvent",
    "FlushObserver",
    "create_patchouli_agent",
]
