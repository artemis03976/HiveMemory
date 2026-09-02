"""Interaction apply 的有界进程内 journal。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

from hivememory.engines.perception.models import TopicMaterializeTask


class InteractionApplyStage(StrEnum):
    """Interaction 从 block+binding 写入到全部后置义务完成的阶段。

    ``INTERACTION_APPLIED`` 对应 P5 的原子 apply 已完成：本轮 ``LogicalBlock``
    与首次 ``TopicAssetBinding`` 已经在同一个 Store 临界区提交，等待本地/后置
    义务（token-overflow compact、状态释放、settlement admission）完成。
    """

    INTERACTION_APPLIED = "interaction_applied"
    LOCAL_COMPLETED = "local_completed"
    COMPLETED = "completed"


@dataclass(frozen=True)
class InteractionApplyRecord:
    """一次 interaction 的进程内 apply 状态。"""

    topic_id: str
    stage: InteractionApplyStage
    input_digest: str | None = None
    settlement_to_submit: TopicMaterializeTask | None = None


class InMemoryInteractionApplyJournal:
    """为 Interaction Submission retry 提供有界的进程内幂等窗口。"""

    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max(1, max_entries)
        self._records: OrderedDict[str, InteractionApplyRecord] = OrderedDict()

    def get(self, interaction_id: str) -> InteractionApplyRecord | None:
        record = self._records.get(interaction_id)
        if record is not None:
            self._records.move_to_end(interaction_id)
        return record

    def record_interaction_applied(
        self,
        interaction_id: str,
        topic_id: str,
        input_digest: str,
    ) -> None:
        """记录原子 apply 已完成；等价 retry 不再触发重复 Store 写入。

        同一 ``interaction_id`` 携带不同 ``input_digest`` 属于不一致 retry，必须
        显式报冲突，避免用新输入覆盖已写入的 block/binding 事实。
        """
        self._require_digest_text(input_digest)
        existing = self._records.get(interaction_id)
        if existing is not None:
            self._require_topic(interaction_id, topic_id, existing)
            self._require_digest(interaction_id, input_digest, existing)
        else:
            self._records[interaction_id] = InteractionApplyRecord(
                topic_id=topic_id,
                stage=InteractionApplyStage.INTERACTION_APPLIED,
                input_digest=input_digest,
            )
        self._touch_and_trim(interaction_id)

    def record_local_completed(
        self,
        interaction_id: str,
        topic_id: str,
        settlement_to_submit: TopicMaterializeTask | None,
    ) -> None:
        existing = self._require_record(interaction_id, topic_id)
        if existing.stage is InteractionApplyStage.COMPLETED:
            return
        self._records[interaction_id] = InteractionApplyRecord(
            topic_id=topic_id,
            stage=InteractionApplyStage.LOCAL_COMPLETED,
            input_digest=existing.input_digest,
            settlement_to_submit=settlement_to_submit,
        )
        self._touch_and_trim(interaction_id)

    def complete(self, interaction_id: str, topic_id: str) -> None:
        existing = self._require_record(interaction_id, topic_id)
        if existing.stage is InteractionApplyStage.INTERACTION_APPLIED:
            raise RuntimeError(
                f"interaction '{interaction_id}' local obligations are not completed"
            )
        self._records[interaction_id] = InteractionApplyRecord(
            topic_id=topic_id,
            stage=InteractionApplyStage.COMPLETED,
            input_digest=existing.input_digest,
        )
        self._touch_and_trim(interaction_id)

    def __len__(self) -> int:
        return len(self._records)

    def _require_record(
        self,
        interaction_id: str,
        topic_id: str,
    ) -> InteractionApplyRecord:
        existing = self._records.get(interaction_id)
        if existing is None:
            raise KeyError(f"interaction '{interaction_id}' has not been applied")
        self._require_topic(interaction_id, topic_id, existing)
        return existing

    @staticmethod
    def _require_topic(
        interaction_id: str,
        topic_id: str,
        existing: InteractionApplyRecord,
    ) -> None:
        if existing.topic_id != topic_id:
            raise ValueError(
                f"interaction '{interaction_id}' was already applied to topic "
                f"'{existing.topic_id}'"
            )

    @staticmethod
    def _require_digest(
        interaction_id: str,
        input_digest: str,
        existing: InteractionApplyRecord,
    ) -> None:
        if existing.input_digest != input_digest:
            raise ValueError(
                f"interaction '{interaction_id}' was already applied with a "
                "different input digest"
            )

    @staticmethod
    def _require_digest_text(input_digest: str) -> None:
        if not isinstance(input_digest, str) or not input_digest.strip():
            raise ValueError("input_digest 不能为空")

    def _touch_and_trim(self, interaction_id: str) -> None:
        self._records.move_to_end(interaction_id)
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)


__all__ = [
    "InMemoryInteractionApplyJournal",
    "InteractionApplyRecord",
    "InteractionApplyStage",
]
