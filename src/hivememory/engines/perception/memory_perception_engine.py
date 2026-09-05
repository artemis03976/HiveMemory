"""无状态的短期记忆摄入算法引擎。

``MemoryPerceptionEngine`` 只承载纯算法：把 ``InteractionPayload`` 归并为
不可变 ``LogicalBlock``、估算 token、判断折叠阈值、选择待折叠 blocks 并生成
摘要。它不持有 Store / Journal / Queue，不实现话题路由与 settle 用例（那是
``PerceptionFamiliar`` 的编排职责），也不导入 ``hivememory.patchouli.*``，
可被其他 runtime 复用、可纯单元测试。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.models import (
    ActionReducer,
    IdentityScope,
    LogicalBlock,
    TurnRecord,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.system.config import SemanticFlowPerceptionConfig
from hivememory.utils.token_estimator import estimate_tokens

if TYPE_CHECKING:
    from hivememory.engines.perception.relay_controller import BaseRelayController


class MemoryPerceptionEngine:
    """无状态的短期记忆摄入与 compact 算法。

    职责：``InteractionPayload`` → ``LogicalBlock`` 的纯函数构造、token 估算、
    compact 阈值判断、待折叠 blocks 选择，以及折叠摘要生成（通过持有的
    ``RelayController``）。

    不职责：不持有 Store / Journal / Queue 等运行时状态；不实现 route /
    apply / settle 编排（Familiar 的用例）；不解释触发原因。
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        relay_controller: BaseRelayController,
    ) -> None:
        self.config = config
        self._relay = relay_controller

    def build_block(
        self,
        payload: InteractionPayload,
        identity_scope: IdentityScope,
    ) -> LogicalBlock:
        """纯函数：把一份交互载荷归并为本轮不可变逻辑块。

        ``block_id``/``created_at`` 由模型默认值生成，是 retry 时重新生成的
        随机标识，不参与 retry 等价性判断。
        """
        actions = ActionReducer.reduce(payload.turn_events)
        turn = TurnRecord(
            identity=identity_scope.actor_identity,
            user_query=payload.user_message,
            rewritten_query=payload.rewritten_query,
            assistant_final_text=payload.assistant_final_text or "",
            turn_events=payload.turn_events,
            actions=actions,
            semantic_traces=payload.mtp_traces,
        )
        total_tokens = (
            estimate_tokens(turn.user_query)
            + estimate_tokens(turn.assistant_final_text)
            + sum(
                estimate_tokens(trace.query or "") + estimate_tokens(trace.target or "")
                for trace in turn.semantic_traces
            )
        )
        return LogicalBlock(
            turn=turn,
            total_tokens=total_tokens,
            worth_saving=payload.worth_saving,
        )

    def should_compact(self, total_tokens: int) -> bool:
        """纯函数：判断话题当前 token 总量是否已溢出折叠阈值。"""
        return total_tokens > self.config.fold_token_threshold

    def select_blocks_to_fold(
        self,
        blocks: tuple[LogicalBlock, ...] | list[LogicalBlock],
        retain_recent: int,
    ) -> list[LogicalBlock]:
        """纯函数：选择需要折叠的旧 blocks（保留最近 ``retain_recent`` 个）。"""
        if retain_recent < 1:
            raise ValueError("retain_recent must be >= 1")
        if len(blocks) <= retain_recent:
            return []
        return list(blocks[:-retain_recent])

    def generate_fold_summary(
        self,
        blocks_to_fold: list[LogicalBlock],
        previous_summary: str,
    ) -> str:
        """生成 Page Folding 摘要，封装为 Engine 的算法能力。"""
        return self._relay.generate_summary(blocks_to_fold, previous_summary)


__all__ = ["MemoryPerceptionEngine"]
