"""Patchouli Runtime 生命周期报告模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicShutdownFlushReport:
    """shutdown 阶段 Topic flush 的运行时报告。

    该报告只供 Patchouli Runtime 的 shutdown drain 和可观测性使用，
    不作为 Topic HTTP API 响应。block 计数表示结算前驻留的原始 block
    数量，不等同于过滤后实际进入生成任务的材料数量。
    """

    settled_topic_ids: tuple[str, ...] = ()
    resident_block_count: int = 0


__all__ = ["TopicShutdownFlushReport"]
