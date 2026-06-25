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
    TopicSettlement,
)
from hivememory.core.protocol.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.core.models import Identity
    from hivememory.engines.perception.models import SemanticBuffer

logger = logging.getLogger(__name__)


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
        super().__init__(*args, **kwargs)

    def set_flush_callback(self, callback: Callable[[List[StreamMessage], FlushReason], None]) -> None:
        """
        设置缓冲区刷新回调函数

        Args:
            callback: 刷新时调用的函数，接收 StreamMessage 列表和 FlushReason 参数
        """
        self.on_flush_callback = callback

    # ========== 感知层原语（供 PerceptionFamiliar 调用） ==========

    @abstractmethod
    async def settle_topic(
        self,
        topic_id: str,
        reason: FlushReason = FlushReason.MANUAL,
        wait_for_completion: bool = False,
    ) -> TopicSettlement:
        """原子话题结算，不含任何策略判断。话题不存在时行为由实现定义。"""

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

    # ========== 抽象接口 ==========



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
    "BasePerceptionLayer",
    "BaseRelayController",
]
