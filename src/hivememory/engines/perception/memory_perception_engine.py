"""无状态的短期记忆摄入算法引擎。

``MemoryPerceptionEngine`` 只承载纯算法：把 ``InteractionPayload`` 归并为
不可变 ``LogicalBlock``、估算 token（与记忆生成转写共用动作段的截断规则）、
判断折叠阈值、选择待折叠 blocks 并生成摘要。它不持有 Store / Journal / Queue，不实现话题路由与 settle 用例（那是
``PerceptionFamiliar`` 的编排职责），也不导入 ``hivememory.patchouli.*``，
可被其他 runtime 复用、可纯单元测试。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.config.patchouli import GenerationTranscriptConfig, SemanticFlowPerceptionConfig
from hivememory.core.models import (
    ActionReducer,
    ActorIdentity,
    LogicalBlock,
    TurnRecord,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.prompts.transcript.actions import render_action_entries
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
        transcript_config: GenerationTranscriptConfig | None = None,
    ) -> None:
        self.config = config
        self._relay = relay_controller
        # 与记忆生成转写同一组上限：块的 token 统计按生成实际看到的材料计算。
        self._transcript_config = transcript_config or GenerationTranscriptConfig()

    def build_block(
        self,
        payload: InteractionPayload,
        from_actor: ActorIdentity,
    ) -> LogicalBlock:
        """纯函数：把一份交互载荷归并为本轮不可变逻辑块。

        ``block_id``/``created_at`` 由模型默认值生成，是 retry 时重新生成的
        随机标识，不参与 retry 等价性判断。
        """
        actions = ActionReducer.reduce(payload.turn_events)
        turn = TurnRecord(
            outcome=payload.outcome,
            identity=from_actor,
            user_query=payload.user_message,
            rewritten_query=payload.rewritten_query,
            assistant_final_text=payload.assistant_final_text or "",
            turn_events=tuple(payload.turn_events),
            actions=tuple(actions),
            semantic_traces=tuple(payload.mtp_traces),
        )
        # 思考与外部工具动作按生成转写的同一套截断规则计入；MTP 轨迹仍只计
        # 查询与目标，只有用户消息、回复与 MTP 动作的轮次统计不变。
        total_tokens = (
            estimate_tokens(turn.user_query)
            + estimate_tokens(turn.assistant_final_text)
            + sum(
                estimate_tokens(trace.query or "") + estimate_tokens(trace.target or "")
                for trace in turn.semantic_traces
            )
            + sum(
                estimate_tokens(entry.text)
                for entry in render_action_entries(turn, self._transcript_config)
                if entry.kind != "trace"
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

    async def generate_fold_summary(
        self,
        blocks_to_fold: list[LogicalBlock],
        previous_summary: str,
    ) -> str:
        """生成 Page Folding 摘要，封装为 Engine 的算法能力。"""
        return await self._relay.generate_summary(blocks_to_fold, previous_summary)


__all__ = ["MemoryPerceptionEngine"]
