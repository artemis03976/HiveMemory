"""Patchouli 公共业务异常。"""

from __future__ import annotations


class TopicSettleAdmissionError(RuntimeError):
    """Topic settle 材料未被 memory generation queue 接纳。"""


class TopicBusyError(RuntimeError):
    """同一 Topic 已由另一写入者（Interaction/compact/settle）占用。

    属于可重试的瞬态冲突：Interaction queue 将其分类为瞬态失败，手动 API
    投影为冲突响应，IDLE/LRU 维护据此跳过或改选候选。
    """


__all__ = ["TopicSettleAdmissionError", "TopicBusyError"]
