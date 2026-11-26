"""Patchouli Topic 生命周期公共契约。

这里的对象属于 Topic 管理业务结果，而不是感知使魔实现细节或 HTTP
响应模型。它们可以沿 Patchouli local bus、GlobalSystemBus 和 system
application service 传递，最终由 server 层投影为 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicSettleResult:
    """手动沉淀并退役话题之后的公共业务结果。

    正常返回即表示话题已经退役。没有 generation task 只表示话题没有还没沉淀的交互，
    不代表沉淀失败；接纳失败通过 :class:`TopicSettleAdmissionError` 表达。
    """

    topic_id: str
    generation_task_id: str | None = None

    @property
    def generation_submitted(self) -> bool:
        """是否已经成功接纳记忆生成任务。"""

        return self.generation_task_id is not None


@dataclass(frozen=True)
class TopicDeleteResult:
    """删除话题之后的公共业务结果：``removed`` 为假表示话题正被占用或不存在，没有删除。"""

    topic_id: str
    removed: bool


__all__ = [
    "TopicDeleteResult",
    "TopicSettleResult",
]
