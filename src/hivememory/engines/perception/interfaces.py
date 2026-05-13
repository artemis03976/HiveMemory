"""
HiveMemory 感知层抽象接口

定义感知层各组件的抽象接口，遵循依赖倒置原则。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Callable, TYPE_CHECKING
from hivememory.core.models import StreamMessage
from hivememory.engines.perception.models import (
    FlushReason,
)
from hivememory.patchouli.protocol.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.core.models import Identity
    from hivememory.engines.perception.models import SemanticBuffer

logger = logging.getLogger(__name__)


class BaseArbiter(ABC):
    """
    灰度仲裁器接口

    职责：
        - 处理语义相似度处于灰度区间（0.40-0.75）的模糊情况
        - 使用更精细的模型判断两个意图是否属于同一任务流
        - 返回是否应该继续当前话题

    判定流程：
        1. 接收上一轮上下文和当前查询
        2. 使用 Reranker/SLM 等模型进行二分类
        3. 返回 YES（继续）或 NO（切分）

    Examples:
        >>> arbiter = RerankerArbiter(reranker_service)
        >>> result = arbiter.should_continue_topic(
        ...     previous_context="写贪吃蛇游戏代码",
        ...     current_query="部署到服务器",
        ...     similarity_score=0.55
        ... )
        >>> # result = True (同一任务流的不同阶段)
    """

    @abstractmethod
    def should_continue_topic(
        self,
        previous_context: str,
        current_query: str,
        similarity_score: float,
    ) -> bool:
        """
        判断是否应该继续当前话题

        Args:
            previous_context: 上一轮对话的上下文摘要
            current_query: 当前的用户查询（rewritten_query）
            similarity_score: 语义相似度分数（可选，用于记录或调整决策）

        Returns:
            bool: True 表示应该继续（吸附），False 表示应该切分
        """
        pass

    def is_available(self) -> bool:
        """
        检查仲裁器是否可用

        Returns:
            bool: 是否可用
        """
        return True


class BasePerceptionLayer(ABC):
    """
    感知层抽象基类

    定义所有类型的 PerceptionLayer 的统一接口。

    实现策略：
        - SemanticFlowPerceptionLayer: 语义流策略（LogicalBlock + 语义吸附 + MMU）

    定时调度由 SystemAsyncScheduler 统一管理，
    本组件只暴露 scan_idle_buffers_once() 供调度器调用。

    Examples:
        >>> perception = SemanticFlowPerceptionLayer()
        >>> perception.ingest_payload(payload)
        >>> result = await perception.manual_trigger()
    """

    def __init__(self, *args, **kwargs):
        """
        基类构造函数。

        注意：使用 *args, **kwargs 以兼容子类的不同构造函数签名。
        """
        super().__init__(*args, **kwargs)
        self._idle_timeout_seconds: int = 900

    def set_flush_callback(self, callback: Callable[[List[StreamMessage], FlushReason], None]) -> None:
        """
        设置缓冲区刷新回调函数

        Args:
            callback: 刷新时调用的函数，接收 StreamMessage 列表和 FlushReason 参数
        """
        self.on_flush_callback = callback

    # ========== 调度器调用接口 ==========

    async def scan_idle_buffers_once(self) -> List[str]:
        """
        扫描并 flush 所有空闲超时的 buffer（供 SystemAsyncScheduler 调用）

        子类应重写此方法以实现具体扫描逻辑。
        默认实现返回空列表。

        Returns:
            List[str]: 被 flush 的 topic_id 列表
        """
        return []

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    @abstractmethod
    def ingest_payload(self, payload: InteractionPayload) -> None:
        """
        摄入 Kernel 递归循环的完整交互载荷

        感知层唯一合法入口。子类必须实现此方法。

        Args:
            payload: Kernel → Perception 的原子传输包
        """
        pass

    # ========== MMU 路由与话题管理 (Phase 4.5) ==========

    def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> None:
        """
        路由到指定话题并摄入载荷 (MMU 模式)

        默认实现：忽略 topic_id，直接调用 ingest_payload。
        SemanticFlowPerceptionLayer 重写此方法实现真正的路由。

        Args:
            topic_id: 目标话题 ID 或 "NEW_TOPIC"
            payload: 原子传输包
        """
        self.ingest_payload(payload)

    def get_active_topics_snapshots(
        self,
        identity: Optional["Identity"] = None,
    ) -> List[Any]:
        """
        获取活跃话题快照列表，供路由决策使用

        默认实现：返回空列表（无多话题快照能力）。
        SemanticFlowPerceptionLayer 重写此方法。
        """
        return []

    # ========== 抽象接口 ==========

    @abstractmethod
    def get_buffer(
        self,
        topic_id: str,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        返回类型: SemanticBuffer

        Args:
            topic_id: 话题 ID

        Returns:
            缓冲区对象，不存在返回 None
        """
        pass

    @abstractmethod
    def clear_buffer(
        self,
        topic_id: str,
    ) -> bool:
        """清理指定的缓冲区"""
        pass

    @abstractmethod
    def list_active_buffers(self) -> List[str]:
        """列出所有活跃的缓冲区 key"""
        pass

    @abstractmethod
    def get_buffer_info(
        self,
        topic_id: str,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            topic_id: 话题 ID

        Returns:
            Dict: 缓冲区信息字典
        """
        pass

    @abstractmethod
    async def manual_trigger(
        self,
        topic_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        手动触发话题结算 (Archive + Compact)

        语义：立即归档 + 生成摘要并保留内存。
        话题不会被驱逐，可以继续接收新的交互。

        Args:
            topic_id: 目标话题 ID。如果为 None，则使用 last_active_topic_id 作为回退。

        Returns:
            Dict: 包含 success, topic_id, message, blocks_archived 的结果字典

        Raises:
            ValueError: 如果 topic_id 未指定且没有 last_active_topic_id
        """
        pass


class BaseRelayController(ABC):
    """
    Token 溢出接力控制器基类

    无状态服务，职责：
        - 为 Page Folding 生成 state_summary (generate_summary)
    """

    @abstractmethod
    def generate_summary(self, blocks_to_fold: List[Any], previous_summary: Optional[str] = None) -> str:
        """生成摘要（抽象方法）"""
        pass

    def create_relay_context(self, summary: str) -> str:
        """创建接力上下文文本"""
        return f"[接力摘要] {summary}" if summary else ""


__all__ = [
    "BaseArbiter",
    "BasePerceptionLayer",
    "BaseRelayController",
]
