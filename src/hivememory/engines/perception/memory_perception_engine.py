"""无状态的短期记忆摄入算法引擎。

``MemoryPerceptionEngine`` 只承载纯算法：把 ``InteractionPayload`` 归并为不可变
``LogicalBlock``、按本轮渲染出的材料估算 token、判断折叠阈值、选择要折叠的前缀。它不持有
Store / Journal / Queue，不写折叠摘要与进展记录（写作者由 ``PerceptionFamiliar`` 持有），不实现
话题路由与沉淀用例（那是 ``PerceptionFamiliar`` 的编排职责），也不导入 ``hivememory.patchouli.*``，
可被其他 runtime 复用、可纯单元测试。
"""

from __future__ import annotations

from hivememory.config.patchouli import GenerationTranscriptConfig, PerceptionEngineConfig
from hivememory.core.models import ActorIdentity, LogicalBlock, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.material import render_turn_text
from hivememory.utils.token_estimator import estimate_tokens


class MemoryPerceptionEngine:
    """无状态的短期记忆摄入与折叠算法。

    职责：``InteractionPayload`` → ``LogicalBlock`` 的纯函数构造、token 估算、
    折叠阈值判断与要折叠的前缀选择。

    不职责：不持有 Store / Journal / Queue 等运行时状态；不写摘要；不实现接收交互、折叠与
    沉淀的编排（Familiar 的用例）；不解释触发原因。
    """

    def __init__(
        self,
        config: PerceptionEngineConfig,
        transcript_config: GenerationTranscriptConfig | None = None,
    ) -> None:
        self.config = config
        # 与记忆生成视图同一个工具结果上限：块的 token 统计按生成实际看到的材料计算。
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
        turn = TurnRecord(
            outcome=payload.outcome,
            identity=from_actor,
            turn_events=tuple(payload.turn_events),
        )
        # 统计的是生成实际看到的整轮材料，折叠阈值因此直接反映生成的输入。
        total_tokens = estimate_tokens(render_turn_text(turn.turn_events, self._transcript_config))
        return LogicalBlock(turn=turn, total_tokens=total_tokens)

    def should_fold(self, total_tokens: int) -> bool:
        """纯函数：判断话题工作视图的 token 总量是否已超过折叠阈值。"""
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


__all__ = ["MemoryPerceptionEngine"]
