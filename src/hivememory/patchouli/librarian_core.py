"""
帕秋莉·馆长本体 (Librarian Core)

定位：思考者与管理者
职责：
    - 接收 Eye 传来的感知信号 (Anchors)
    - 维护 Buffer 和漂移检测
    - 调用 Generation 引擎写书
    - 调用 Lifecycle 引擎修书

基于原 PatchouliAgent (agents/patchouli.py) 改造，专注于 Cold Path 处理

作者: HiveMemory Team
版本: 2.0
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING
 
from hivememory.core.models import MemoryAtom, FlushReason, Identity
from hivememory.engines.generation.models import ConversationMessage
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.infrastructure.llm.base import BaseLLMService
from hivememory.infrastructure.embedding.base import BaseEmbeddingService
from hivememory.patchouli.protocol.models import Observation

if TYPE_CHECKING:
    from hivememory.patchouli.config import MemoryPerceptionConfig, MemoryGenerationConfig, MemoryLifecycleConfig

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


class LibrarianCore:
    """
    帕秋莉·馆长本体 (Librarian Core)

    这是帕秋莉的本体，坐在书桌前，一边喝红茶一边处理堆积如山的借阅记录。

    性格特征 (Persona):
        - 极度挑剔: 只记录有价值的信息,拒绝噪音
        - 结构化强迫症: 致力于将混乱的对话转化为规范的JSON
        - 客观中立: 以第三人称视角记录事实

    特性：
        - 异步非阻塞
        - 高智商
        - SOTA 模型驱动

    职责:
        1. 接收 Eye 传来的感知信号 (Anchors)
        2. 维护 Buffer 和漂移检测
        3. 调用 Generation 引擎写书
        4. 调用 Lifecycle 引擎修书 (未来)

    使用示例:
        >>> from hivememory.patchouli.librarian_core import LibrarianCore
        >>> from hivememory.infrastructure.storage import QdrantMemoryStore
        >>>
        >>> storage = QdrantMemoryStore()
        >>> core = LibrarianCore(storage=storage)
        >>>
        >>> # 添加消息到感知层
        >>> core.add_message("user", "帮我写贪吃蛇游戏", "user123", "agent1")
        >>> core.add_message("assistant", "好的，我来实现...", "user123", "agent1")
    """

    def __init__(
        self,
        storage: QdrantMemoryStore = None,
        llm_service: BaseLLMService = None,
        embedding_service: Optional[BaseEmbeddingService] = None,
        perception_config: Optional["MemoryPerceptionConfig"] = None,
        generation_config: Optional["MemoryGenerationConfig"] = None,
        lifecycle_config: Optional["MemoryLifecycleConfig"] = None,
    ):
        """
        初始化馆长本体

        Args:
            storage: Qdrant 存储实例
            llm_service: LLM 服务实例
            embedding_service: Embedding 服务实例（用于感知层语义相似度计算）
            perception_config: 感知层配置
            generation_config: 记忆生成配置
            lifecycle_config: 记忆生命周期配置

        Examples:
            >>> # 使用完整配置（推荐）
            >>> from hivememory.patchouli.config import load_app_config
            >>> config = load_app_config()
            >>> core = LibrarianCore(config=config)
            >>>
            >>> # 使用环境变量默认配置
            >>> core = LibrarianCore()
            >>>
            >>> # 使用简单感知层（覆盖配置）
            >>> from hivememory.patchouli.config import MemoryPerceptionConfig
            >>> perception_config = MemoryPerceptionConfig(layer_type="simple")
            >>> core = LibrarianCore(perception_config=perception_config)
        """
        if perception_config is None:
            from hivememory.patchouli.config import MemoryPerceptionConfig
            perception_config = MemoryPerceptionConfig()
        if generation_config is None:
            from hivememory.patchouli.config import MemoryGenerationConfig
            generation_config = MemoryGenerationConfig()
        if lifecycle_config is None:
            from hivememory.patchouli.config import MemoryLifecycleConfig
            lifecycle_config = MemoryLifecycleConfig()

        self.storage = storage
        self.llm_service = llm_service
        self.embedding_service = embedding_service

        self.perception_config = perception_config
        self.generation_config = generation_config
        self.lifecycle_config = lifecycle_config
        
        self.perception_layer: Optional[BasePerceptionLayer] = None
        self.generation_orchestrator: Optional[MemoryGenerationOrchestrator] = None
        self.lifecycle_manager: Optional[LifecycleManager] = None

        # 统一使用工厂函数初始化各模块
        self._init_modules()

        # Flush 事件观察者列表
        self._flush_observers: List[FlushObserver] = []

        # 记录实际使用的感知层类型
        layer_type = getattr(self.perception_config, "layer_type", "semantic_flow") if self.perception_config else "semantic_flow"
        logger.info(f"LibrarianCore 初始化完成 (perception_layer_type={layer_type})")

    def _init_modules(self) -> None:
        """
        统一初始化所有子模块

        使用工厂函数创建各模块，确保初始化模式一致：
        - perception: create_default_perception_layer()
        - generation: create_default_generation_orchestrator()
        - lifecycle: create_default_lifecycle_manager()
        """
        # 初始化感知层
        from hivememory.engines.perception import create_default_perception_layer
        self.perception_layer = create_default_perception_layer(
            config=self.perception_config,
            embedding_service=self.embedding_service,
            on_flush_callback=self._on_perception_flush,
        )
        logger.debug(f"{type(self.perception_layer).__name__} 已初始化")

        # 初始化记忆生成编排器
        from hivememory.engines.generation import create_default_generation_orchestrator
        self.generation_orchestrator = create_default_generation_orchestrator(
            storage=self.storage,
            llm_service=self.llm_service,
            config=self.generation_config,
        )
        logger.debug("MemoryGenerationOrchestrator 已初始化")

        # 初始化记忆生命周期管理器
        from hivememory.engines.lifecycle import create_default_lifecycle_manager
        self.lifecycle_manager = create_default_lifecycle_manager(
            storage=self.storage,
            config=self.lifecycle_config,
        )
        logger.debug("LifecycleManager 已初始化")

    # ========== 感知层 API ==========

    def perceive(
        self,
        observation: Observation,
    ) -> None:
        """
        统一感知入口 (Cold Path)

        无论是经过 Eye 处理的用户查询，还是普通的 Assistant/System 消息，
        都通过此接口进入感知层。参数统一为 Observation 对象。

        Args:
            observation: 感知信号对象
        """
        # 1. 从 Observation 创建 Identity
        identity = observation.identity

        # 2. 提取参数
        role = observation.role
        content = observation.raw_message
        rewritten_query = observation.anchor
        gateway_intent = observation.gateway_context.get("intent")
        worth_saving = observation.gateway_context.get("worth_saving")

        if rewritten_query:
            logger.debug(
                f"LibrarianCore 接收到 Eye 信号: anchor='{rewritten_query[:20]}...', "
                f"intent={gateway_intent}"
            )
        else:
            logger.debug(f"LibrarianCore 接收到普通消息: role={role}")

        # 3. 添加到感知层
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.add_message(
            role=role,
            content=content,
            identity=identity,
            rewritten_query=rewritten_query,
            gateway_intent=gateway_intent,
            worth_saving=worth_saving,
        )

        logger.debug(f"向感知层添加消息: {role} - {content[:50]}...")

    def flush_perception(
        self,
        identity: Identity,
    ) -> None:
        """
        手动触发感知层 Flush

        Args:
            identity: 身份标识对象

        Examples:
            >>> from hivememory.core.models import Identity
            >>> identity = Identity(user_id="user123", agent_id="chatbot", session_id="session_456")
            >>> core.flush_perception(identity)
        """
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.flush_buffer(identity=identity)

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

            # 获取正确的 user_id 和 agent_id
            first_msg = messages[0]
            user_id = getattr(first_msg, "user_id", "unknown")
            agent_id = getattr(first_msg, "agent_id", "librarian_core")

            logger.info(f"馆长本体开始处理 {len(messages)} 条消息...")

            # 调用编排器处理
            memories = self.generation_orchestrator.process(
                messages=messages,
            )

            # 创建 Flush 事件并通知观察者
            event = FlushEvent(
                messages=messages,
                reason=reason,
                memories=memories or [],
            )
            self._notify_flush_observers(event)

            logger.info(f"馆长本体处理完成")
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
            >>> core.add_flush_observer(my_observer)
        """
        self._flush_observers.append(observer)
        logger.debug(f"添加 Flush 观察者，当前共 {len(self._flush_observers)} 个")

    def remove_flush_observer(self, observer: FlushObserver) -> None:
        """
        移除 Flush 事件观察者

        Args:
            observer: 要移除的回调函数

        Examples:
            >>> core.remove_flush_observer(my_observer)
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
        identity: Identity,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        Args:
            identity: 身份标识对象

        Returns:
            Buffer 实例（SimpleBuffer 或 SemanticBuffer）
        """
        if not self.perception_layer:
            return None

        return self.perception_layer.get_buffer(identity=identity)

    def clear_buffer(self, identity: Identity) -> bool:
        """
        清理指定的 Buffer

        Args:
            identity: 身份标识对象

        Returns:
            bool: 是否成功清理

        Examples:
            >>> from hivememory.core.models import Identity
            >>> identity = Identity(user_id="user123", agent_id="chatbot", session_id="session_456")
            >>> core.clear_buffer(identity)
            True
        """
        if not self.perception_layer:
            return False

        return self.perception_layer.clear_buffer(identity=identity)

    def get_buffer_info(self, identity: Identity) -> Dict[str, Any]:
        """
        获取 Buffer 信息（统一接口）

        Args:
            identity: 身份标识对象

        Returns:
            Dict: Buffer 信息字典

        Examples:
            >>> from hivememory.core.models import Identity
            >>> identity = Identity(user_id="user123", agent_id="chatbot", session_id="session_456")
            >>> info = core.get_buffer_info(identity)
            >>> print(f"消息数量: {info.get('block_count', info.get('message_count', 0))}")
        """
        if not self.perception_layer:
            return {"exists": False, "mode": "none"}

        return self.perception_layer.get_buffer_info(identity=identity)

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表

        Examples:
            >>> buffers = core.list_active_buffers()
            >>> print(f"当前有 {len(buffers)} 个活跃 Buffer")
        """
        if not self.perception_layer:
            return []

        return self.perception_layer.list_active_buffers()

    # ========== 生命周期管理 API (未来扩展) ==========

    def start_gardening(self):
        """
        开启定时维护模式

        未来实现：调用 LifecycleManager 进行定期维护
        """
        # TODO: 实现定时维护模式
        logger.info("定时维护模式尚未实现")


# 便捷函数
def create_librarian_core(
    storage: Optional[QdrantMemoryStore] = None,
    layer_type: str = "semantic_flow",
) -> LibrarianCore:
    """
    创建馆长本体实例

    Args:
        storage: Qdrant 存储实例（可选）
        layer_type: 感知层类型，"semantic_flow" 或 "simple"（默认 "semantic_flow"）

    Returns:
        LibrarianCore: 实例

    Examples:
        >>> # 使用语义流感知层（默认）
        >>> core = create_librarian_core()
        >>>
        >>> # 使用简单感知层
        >>> core = create_librarian_core(layer_type="simple")
    """
    from hivememory.patchouli.config import MemoryPerceptionConfig

    config = MemoryPerceptionConfig(layer_type=layer_type)
    return LibrarianCore(storage=storage, perception_config=config)


__all__ = [
    "LibrarianCore",
    "FlushEvent",
    "FlushObserver",
    "create_librarian_core",
]
