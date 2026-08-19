"""
HiveMemory 感知层抽象接口

定义感知层各组件的抽象接口，遵循依赖倒置原则。

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 2.0.0
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple, TYPE_CHECKING
from hivememory.engines.perception.models import (
    FlushReason,
    TopicMaterializeTask,
)
from hivememory.core.protocol.models import InteractionPayload

if TYPE_CHECKING:
    from hivememory.core.models import WorkspaceAccessContext, WorkspaceTopicKey

logger = logging.getLogger(__name__)


class BasePerceptionLayer(ABC):
    """
    感知层抽象基类

    定义所有类型的 PerceptionLayer 的统一接口。

    实现策略：
        - SemanticFlowPerceptionLayer: 语义流策略（LogicalBlock + MMU）

    定时调度由 SystemAsyncScheduler 统一管理

    Examples:
        >>> perception = SemanticFlowPerceptionLayer()
        >>> perception.ingest_payload(payload)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ========== 感知层原语（供 PerceptionFamiliar 调用） ==========

    @abstractmethod
    async def settle_topic(
        self,
        topic_key: "WorkspaceTopicKey",
        reason: FlushReason = FlushReason.MANUAL,
    ) -> Optional[TopicMaterializeTask]:
        """原子话题结算，不含任何策略判断。话题不存在或为空时返回 None。"""

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    @abstractmethod
    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
        interaction_id: str | None = None,
    ) -> Optional[TopicMaterializeTask]:
        """
        摄入完整交互载荷。

        Returns:
            如发生 TOKEN_OVERFLOW 结算则返回 TopicMaterializeTask，否则 None
        """

    # ========== MMU 路由与话题管理 (Phase 4.5) ==========

    @abstractmethod
    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
        interaction_id: str | None = None,
    ) -> Tuple[str, Optional[TopicMaterializeTask]]:
        """
        路由到指定话题并摄入载荷。

        Returns:
            (real_topic_id, TopicMaterializeTask | None)
        """

    @abstractmethod
    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        access_context: "WorkspaceAccessContext",
    ) -> str:
        """确保目标短期话题存在，并返回真实 topic_id。"""

    @abstractmethod
    def swap_out_topic(self, topic_key: "WorkspaceTopicKey") -> bool:
        """显式换出指定话题，不触发结算。返回是否存在该话题。"""


class BaseRelayController(ABC):
    """
    Token 溢出接力控制器基类

    无状态服务，职责：
        - 为 Page Folding 生成 state_summary (generate_summary)
    """

    @abstractmethod
    def generate_summary(self, blocks_to_fold: List[Any], previous_summary: Optional[str] = None) -> str:
        """生成摘要（抽象方法）"""

    def create_relay_context(self, summary: str) -> str:
        """创建接力上下文文本"""
        return f"[接力摘要] {summary}" if summary else ""


__all__ = [
    "BasePerceptionLayer",
    "BaseRelayController",
]
