"""Patchouli Runtime 生命周期报告模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RestoredTopicPoolReport:
    """启动时从话题日志读回话题池的结果，只供运行时观测，不作为 HTTP 响应。

    ``restored`` 是回到话题池的话题数；``retired`` 是已经退役、最后一次沉淀没有完成、在生成
    队列启动之后重新发起这次沉淀的话题数；``rejected`` 是读不回、移到一旁的日志数。
    """

    restored: int = 0
    retired: int = 0
    rejected: int = 0


__all__ = ["RestoredTopicPoolReport"]
