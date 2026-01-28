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
版本: 2.1
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING

from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.models import FlushEvent, FlushReason
from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.patchouli.protocol.models import Observation

FlushObserver = Callable[[FlushEvent], None]

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.engines.lifecycle.engine import MemoryLifecycleEngine

logger = logging.getLogger(__name__)


class LibrarianCore:
    """
    帕秋莉·馆长本体 (Librarian Core)

    这是帕秋莉的本体，坐在书桌前，一边喝红茶一边处理堆积如山的借阅记录。

    遵循显式依赖注入原则：所有子组件必须通过构造函数传入，
    不在内部实例化依赖项。由 PatchouliSystem 负责组装和注入。

    职责:
        1. 接收 Eye 传来的感知信号 (Anchors)
        2. 维护 Buffer 和漂移检测
        3. 调用 Generation 引擎写书
        4. 调用 Lifecycle 引擎修书

    使用示例:
        >>> # 推荐：通过 PatchouliSystem 使用
        >>> from hivememory.patchouli import PatchouliSystem
        >>> system = PatchouliSystem()
        >>> core = system.librarian_core
        >>>
        >>> # 高级：手动注入组件
        >>> core = LibrarianCore(
        ...     storage=storage,
        ...     perception_layer=perception_layer,
        ...     generation_engine=generation_engine,
        ...     lifecycle_engine=lifecycle_engine,
        ... )
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        generation_engine: MemoryGenerationEngine,
        perception_layer: BasePerceptionLayer,
        lifecycle_engine: MemoryLifecycleEngine,
    ):
        """
        初始化馆长本体

        Args:
            storage: Qdrant 存储实例
            perception_layer: 感知层实例（预构建，由 PatchouliSystem 注入）
            generation_engine: 记忆生成引擎（预构建，由 PatchouliSystem 注入）
            lifecycle_engine: 记忆生命周期引擎（预构建，由 PatchouliSystem 注入）

        Note:
            推荐通过 PatchouliSystem 使用，它会自动构建并注入所有组件。
        """
        self.storage = storage

        # 使用注入的组件
        self.perception_layer = perception_layer
        self.generation_engine = generation_engine
        self.lifecycle_engine = lifecycle_engine

        # 设置感知层的 flush 回调
        self.perception_layer.set_flush_callback(self._on_perception_flush)

        # Flush 事件观察者列表
        self._flush_observers: List[FlushObserver] = []

        # 记录实际使用的感知层类型
        layer_type = type(self.perception_layer).__name__ if self.perception_layer else "None"
        logger.info(f"LibrarianCore 初始化完成 (perception_layer={layer_type})")

    def add_flush_observer(self, observer: FlushObserver) -> None:
        """添加 Flush 事件观察者"""
        self._flush_observers.append(observer)

    def remove_flush_observer(self, observer: FlushObserver) -> None:
        """移除 Flush 事件观察者"""
        if observer in self._flush_observers:
            self._flush_observers.remove(observer)

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

        self.perception_layer.perceive(
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
        self.perception_layer.flush_buffer(identity=identity)

    def _on_perception_flush(
        self,
        messages: List[StreamMessage],
        reason: FlushReason,
    ) -> None:
        """
        感知层 Flush 回调（统一接口）
        将感知层生成的消息传递给编排器处理

        Args:
            messages: StreamMessage 列表
            reason: Flush 原因
        """
        try:
            # 从消息中提取上下文
            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            logger.info(f"LibrarianCore 开始处理 {len(messages)} 条消息...")

            # 调用生成引擎处理
            memories = self.generation_engine.process(
                messages=messages,
            )

            logger.info(f"馆长本体处理完成")
            if memories:
                logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

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

        未来实现：调用 MemoryLifecycleEngine 进行定期维护
        """
        # TODO: 实现定时维护模式
        logger.info("定时维护模式尚未实现")


__all__ = [
    "LibrarianCore",
]
