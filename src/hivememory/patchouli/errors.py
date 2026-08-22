"""Patchouli 公共业务异常。"""

from __future__ import annotations


class TopicSettleAdmissionError(RuntimeError):
    """Topic settle 材料未被 memory generation queue 接纳。"""


__all__ = ["TopicSettleAdmissionError"]
