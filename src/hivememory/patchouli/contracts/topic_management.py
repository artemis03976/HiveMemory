"""Patchouli Topic 生命周期公共契约。

这里的对象属于 Topic 管理业务结果，而不是感知使魔实现细节或 HTTP
响应模型。它们可以沿 Patchouli local bus、GlobalSystemBus 和 system
application service 传递，最终由 server 层投影为 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicSettleResult:
    """Topic settle 完成后的公共业务结果。

    正常返回即表示 Topic 生命周期已经结束。没有 generation task 只表示
    当前没有可提交的结算材料，不代表 settle 失败；接纳失败通过
    :class:`TopicSettleAdmissionError` 表达。
    """

    topic_id: str
    generation_task_id: str | None = None

    @property
    def generation_submitted(self) -> bool:
        """是否已经成功接纳记忆生成任务。"""

        return self.generation_task_id is not None


@dataclass(frozen=True)
class TopicEvictionResult:
    """Topic 从活跃池移除后的公共业务结果。"""

    topic_id: str
    removed: bool


__all__ = [
    "TopicEvictionResult",
    "TopicSettleResult",
]
