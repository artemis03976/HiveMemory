"""单帧上下文的估算与有类型的超限错误，不参与生成参数透传。"""

from collections.abc import Sequence

from hivememory.utils.token_estimator import TokenEstimator

# 每条消息的角色与分隔符开销；增量估算请求时须与整体估算使用同一值。
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_context_tokens(messages: Sequence[dict[str, str]]) -> int:
    """沿用文本估算，并为每条消息的角色与分隔符预留固定开销。"""
    return sum(
        TokenEstimator.estimate(message.get("content", "")) + MESSAGE_OVERHEAD_TOKENS
        for message in messages
    )


class InputTooLargeError(Exception):
    """固定提示词与当前用户输入已超出预算，压缩历史无法解决。"""


class TurnContextBudgetExceededError(Exception):
    """下一次生成的完整上下文已超出本帧输入上限。"""

    def __init__(self, estimated_tokens: int, input_token_limit: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.input_token_limit = input_token_limit
        super().__init__(
            f"Frame context exceeds input budget: {estimated_tokens} > {input_token_limit}"
        )
