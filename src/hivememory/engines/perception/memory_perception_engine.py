"""无状态的短期记忆摄入算法引擎。

``MemoryPerceptionEngine`` 只承载纯算法：把 ``InteractionPayload`` 归并为
不可变 ``LogicalBlock``、按本轮渲染出的材料估算 token、
判断折叠阈值、选择待折叠 blocks 并生成摘要，以及写出进展记录的一条。它不持有 Store / Journal /
Queue，不实现话题路由与 settle 用例（那是
``PerceptionFamiliar`` 的编排职责），也不导入 ``hivememory.patchouli.*``，
可被其他 runtime 复用、可纯单元测试。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from hivememory.config.patchouli import GenerationTranscriptConfig, PerceptionEngineConfig
from hivememory.core.models import ActorIdentity, LogicalBlock, TopicProgressEntry, TurnRecord
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.material import render_turn_text
from hivememory.utils.token_estimator import estimate_tokens

if TYPE_CHECKING:
    from hivememory.engines.perception.progress_record import ProgressRecordWriter
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
        config: PerceptionEngineConfig,
        relay_controller: BaseRelayController,
        transcript_config: GenerationTranscriptConfig | None = None,
        progress_writer: ProgressRecordWriter | None = None,
    ) -> None:
        self.config = config
        self._relay = relay_controller
        # 与记忆生成视图同一个工具结果上限：块的 token 统计按生成实际看到的材料计算。
        self._transcript_config = transcript_config or GenerationTranscriptConfig()
        # 没有可用的模型时为 None：不写进展记录。
        self._progress_writer = progress_writer

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

    @property
    def writes_progress_record(self) -> bool:
        """是否有写进展记录的模型；没有时话题不写进展记录。"""
        return self._progress_writer is not None

    async def generate_progress_entry(
        self,
        previous_record: Sequence[TopicProgressEntry],
        blocks: Sequence[LogicalBlock],
        *,
        first_turn_number: int,
    ) -> str:
        """以旧进展记录与还没整理的交互写出新的一条的正文；没有写进展记录的模型时是调用错误。"""
        if self._progress_writer is None:
            raise RuntimeError("没有写进展记录的模型")
        return await self._progress_writer.write_entry(
            previous_record, blocks, first_turn_number=first_turn_number
        )


__all__ = ["MemoryPerceptionEngine"]
