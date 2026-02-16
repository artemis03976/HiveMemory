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

from hivememory.core.models import Identity, MemoryAtom, StreamMessage
from hivememory.engines.perception.models import FlushEvent, FlushReason
from hivememory.engines.generation.models import GenerationRequest, WriteFocus, UpdateFocus
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
    不在内部实例化依赖项。由 PatchouliKernel 负责组装和注入。

    职责:
        1. 接收 Eye 传来的感知信号 (Anchors)
        2. 维护 Buffer 和漂移检测
        3. 调用 Generation 引擎写书
        4. 调用 Lifecycle 引擎修书

    使用示例:
        >>> # 推荐：通过 PatchouliKernel 使用
        >>> from hivememory.patchouli import PatchouliKernel
        >>> kernel = PatchouliKernel()
        >>> core = kernel.librarian_core
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
            perception_layer: 感知层实例（预构建，由 PatchouliKernel 注入）
            generation_engine: 记忆生成引擎（预构建，由 PatchouliKernel 注入）
            lifecycle_engine: 记忆生命周期引擎（预构建，由 PatchouliKernel 注入）

        Note:
            推荐通过 PatchouliKernel 使用，它会自动构建并注入所有组件。
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
        worth_saving = observation.worth_saving

        if rewritten_query:
            logger.debug(
                f"LibrarianCore 接收到 Eye 信号: anchor='{rewritten_query[:20]}...', "
                f"worth_saving={worth_saving}"
            )
        else:
            logger.debug(f"LibrarianCore 接收到普通消息: role={role}")

        self.perception_layer.perceive(
            role=role,
            content=content,
            identity=identity,
            rewritten_query=rewritten_query,
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
            # 双重处理防护: MTP_WRITE/MTP_UPDATE 由专用 handler 直接处理
            if reason == FlushReason.MTP_WRITE:
                logger.debug("MTP_WRITE flush，跳过 Mode A 回调 (由 handle_write_signal 处理)")
                return

            if reason == FlushReason.MTP_UPDATE:
                logger.debug("MTP_UPDATE flush，跳过 Mode A 回调 (由 handle_update_signal 处理)")
                return

            # 从消息中提取上下文
            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            logger.info(f"LibrarianCore 开始处理 {len(messages)} 条消息...")

            # 调用生成引擎处理
            memories = self.generation_engine.process(
                GenerationRequest(context_messages=messages),
            )

            logger.info(f"帕秋莉处理完成")
            if memories:
                logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

    # ========== WRITE 指令处理 ==========

    def handle_write_signal(self, write_focus: WriteFocus) -> List[MemoryAtom]:
        """
        处理 MTP WRITE 指令信号

        流程:
            1. 强制刷新感知层 buffer (reason=MTP_WRITE)
            2. 将 buffer 消息 + WriteFocus 打包为 GenerationRequest
            3. 调用 Generation Engine (Mode B) 处理

        Args:
            write_focus: WRITE 指令的聚焦内容

        Returns:
            List[MemoryAtom]: 生成的记忆原子列表
        """
        identity = write_focus.identity

        # Step 1: 强制刷新 buffer，获取积压的对话上下文
        buffer_messages = self.perception_layer.flush_buffer(
            identity=identity,
            reason=FlushReason.MTP_WRITE,
        )

        logger.info(
            f"WRITE 信号处理: flush 获取 {len(buffer_messages)} 条上下文消息"
        )

        # Step 2: 打包 GenerationRequest (Mode B)
        request = GenerationRequest(
            context_messages=buffer_messages,
            write_focus=write_focus,
        )

        # Step 3: 调用 Generation Engine
        try:
            memories = self.generation_engine.process(request)
            if memories:
                logger.info(f"WRITE 信号处理完成: 生成 {len(memories)} 条记忆")
            else:
                logger.warning("WRITE 信号处理完成: 未生成记忆")
            return memories
        except Exception as e:
            logger.error(f"WRITE 信号处理失败: {e}", exc_info=True)
            return []

    # ========== UPDATE 指令处理 ==========

    def handle_update_signal(self, update_focus: UpdateFocus) -> List[MemoryAtom]:
        """
        处理 MTP UPDATE 指令信号

        流程:
            1. 强制刷新感知层 buffer (reason=MTP_UPDATE)
            2. 从 storage 加载目标记忆
            3. 注入 existing_memory 到 update_focus
            4. 打包 GenerationRequest (Mode C)
            5. 调用 Generation Engine 处理

        Args:
            update_focus: UPDATE 指令的聚焦内容

        Returns:
            List[MemoryAtom]: 更新后的记忆原子列表

        Raises:
            ValueError: 目标记忆不存在时抛出
        """
        identity = update_focus.identity

        # Step 1: 强制刷新 buffer，获取积压的对话上下文
        buffer_messages = self.perception_layer.flush_buffer(
            identity=identity,
            reason=FlushReason.MTP_UPDATE,
        )

        logger.info(
            f"UPDATE 信号处理: flush 获取 {len(buffer_messages)} 条上下文消息, "
            f"alias='{update_focus.target_alias}'"
        )

        # Step 2: 从 storage 加载目标记忆
        from uuid import UUID as _UUID
        existing = self.storage.get_memory(_UUID(update_focus.target_uuid))
        if existing is None:
            raise ValueError(
                f"Memory {update_focus.target_uuid} not found in storage"
            )

        # Step 3: 注入 existing_memory
        update_focus.existing_memory = existing

        # Step 4: 打包 GenerationRequest (Mode C)
        request = GenerationRequest(
            context_messages=buffer_messages,
            update_focus=update_focus,
        )

        # Step 5: 调用 Generation Engine
        try:
            memories = self.generation_engine.process(request)
            if memories:
                logger.info(f"UPDATE 信号处理完成: 更新 {len(memories)} 条记忆")
            else:
                logger.warning("UPDATE 信号处理完成: 未更新记忆")
            return memories
        except Exception as e:
            logger.error(f"UPDATE 信号处理失败: {e}", exc_info=True)
            return []

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
        获取 Buffer 信息

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
        logger.warning("定时维护模式尚未实现")


__all__ = [
    "LibrarianCore",
]
