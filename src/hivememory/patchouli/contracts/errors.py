"""Patchouli 资源解析、话题结算与交互提交的公共错误。"""

from typing import Literal

from hivememory.components.work_queue import WorkState
from hivememory.core.errors import DomainError


class ProfileResolutionError(DomainError):
    """已找到的 Agent Profile 类型或配置不满足执行要求。"""

    code = "patchouli.profile.resolution_failed"

    def __init__(
        self, *, agent_alias: str, reason: Literal["type_mismatch", "invalid_config"]
    ) -> None:
        super().__init__(details={"agent_alias": agent_alias, "reason": reason})


class TopicSettleAdmissionError(DomainError):
    """Topic settle 材料未被 memory generation queue 接纳。"""

    code = "patchouli.topic.settle_rejected"


class TopicBusyError(DomainError):
    """同一 Topic 已由另一写入者（Interaction/compact/settle）占用。

    属于可重试的瞬态冲突：Interaction queue 将其分类为瞬态失败，手动 API
    投影为冲突响应，IDLE/LRU 维护据此跳过或改选候选。
    """

    code = "patchouli.topic.busy"


class InteractionAdmissionUnavailableError(DomainError):
    """交互提交暂未被摄入队列接纳：容量已满或队列已停止。

    属于可重试的拥塞：队列没有接管这次提交，调用方以同一 interaction_id
    稍后重发即可，不会形成重复交互。
    """

    code = "patchouli.interaction.admission_unavailable"

    def __init__(self, *, reason: Literal["capacity_rejected", "queue_stopped"]) -> None:
        super().__init__(details={"reason": reason})


ActiveFinalizationStage = Literal[
    "interaction_admission",
    "interaction_apply",
]


class ActiveInteractionFinalizationError(DomainError):
    """Active interaction 未能跨过 admission/apply 硬成功边界。"""

    code = "patchouli.interaction.finalization_failed"

    def __init__(
        self,
        *,
        interaction_id: str,
        stage: ActiveFinalizationStage,
        reason: str,
        work_state: WorkState | None = None,
        error_class: str | None = None,
    ) -> None:
        self.interaction_id = interaction_id
        self.stage = stage
        self.reason = reason
        self.work_state = work_state
        self.error_class = error_class
        super().__init__(f"Active interaction finalization failed at {stage}: {reason}")
