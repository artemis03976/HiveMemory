"""Interaction apply 的有界进程内 journal。

journal 同时拥有 apply 指纹（digest）的构造与校验：同一 ``interaction_id``
的 retry 必须携带等价输入，指纹只覆盖可稳定重建的 canonical 输入。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum

from hivememory.core.models import IdentityScope, LogicalBlock
from hivememory.engines.perception.models import TopicMaterializeTask


def compute_apply_digest(
    block: LogicalBlock,
    asset_id_and_refs,
    model_used: str | None,
    identity_scope: IdentityScope,
) -> str:
    """计算一次 Interaction apply 的稳定输入摘要。

    digest 只覆盖可稳定重建的 canonical 输入（block 事实、binding refs 与
    参与 apply 的 metadata），刻意排除 ``block_id``/``created_at``/``bound_at``
    等 retry 时重新生成的随机或时钟值。它只用于判断同一 ``interaction_id``
    的 retry 是否等价，不可作为 used refs 或 settlement 查询依据。
    """
    turn_dump = block.turn.model_dump(mode="json")
    # turn_id 与 block_id/created_at 一样，是 retry 时重新生成的随机标识。
    turn_dump.pop("turn_id", None)
    canonical = {
        # Workspace 是 Store apply 的寻址边界；只依赖 block 内的 actor identity
        # 会把同一 interaction 在不同 Workspace 的提交误判为等价 retry。
        "identity_scope": identity_scope.model_dump(mode="json"),
        "turn": turn_dump,
        "total_tokens": block.total_tokens,
        "worth_saving": block.worth_saving,
        "gateway_intent": block.gateway_intent,
        "model_used": model_used or "",
        "asset_refs": sorted(
            (asset_id, asset_ref.token) for asset_id, asset_ref in asset_id_and_refs
        ),
    }
    payload = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
                f"interaction '{interaction_id}' was already applied to topic '{existing.topic_id}'"
            )

    @staticmethod
    def _require_digest(
        interaction_id: str,
        input_digest: str,
        existing: InteractionApplyRecord,
    ) -> None:
        if existing.input_digest != input_digest:
            raise ValueError(
                f"interaction '{interaction_id}' was already applied with a different input digest"
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
    "compute_apply_digest",
]
