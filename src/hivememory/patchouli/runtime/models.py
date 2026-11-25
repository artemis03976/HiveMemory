"""Patchouli Runtime 生命周期报告模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicShutdownFlushReport:
    """shutdown 阶段 Topic flush 的运行时报告。

    该报告只供 Patchouli Runtime 的 shutdown drain 和可观测性使用，
    不作为 Topic HTTP API 响应。block 计数表示处理前驻留的工作视图 block
    数量，不等同于实际进入生成任务的材料数量。

    关闭时只沉淀、不退役：``settled_topic_ids`` 是这次沉淀了新交互的话题，它们留在
    话题池里随快照保存；``retired_topic_ids`` 是从没接收过交互、直接退役的空话题。
    接收过交互、已经全部沉淀的话题不出现在报告中。

    ``failed_topic_ids`` 记录没能沉淀的 Topic（正被占用、admission 失败等）：它们带着
    未沉淀的交互照常保存，单个 Topic 的失败不会阻止其余 Topic 的处理。
    """

    settled_topic_ids: tuple[str, ...] = ()
    retired_topic_ids: tuple[str, ...] = ()
    resident_block_count: int = 0
    failed_topic_ids: tuple[str, ...] = ()


__all__ = ["TopicShutdownFlushReport"]
