"""Patchouli Runtime 生命周期报告模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicShutdownFlushReport:
    """shutdown 阶段 Topic flush 的运行时报告。

    该报告只供 Patchouli Runtime 的 shutdown drain 和可观测性使用，
    不作为 Topic HTTP API 响应。block 计数表示结算前驻留的原始 block
    数量，不等同于过滤后实际进入生成任务的材料数量。

    ``generation_skipped_topic_ids`` 是 ``settled_topic_ids`` 的子集：这些
    Topic 已正常结束生命周期，但没有建立 generation task。它不表示异常。

    ``failed_topic_ids`` 记录未能完成统一 settle 协议的 Topic（begin 时 busy、
    admission 失败等）。这些 Topic 通过 abort 恢复 IDLE 并保留内容，不计入
    ``settled_topic_ids``；单个 Topic 的失败不会阻止其余 Topic 的清理。
    """

    settled_topic_ids: tuple[str, ...] = ()
    generation_skipped_topic_ids: tuple[str, ...] = ()
    resident_block_count: int = 0
    failed_topic_ids: tuple[str, ...] = ()


__all__ = ["TopicShutdownFlushReport"]
