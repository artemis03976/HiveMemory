"""Gateway 内部能力适配错误与对用户选择的受控拒绝。"""

from hivememory.core.errors import WorkspaceDomainError


class RecoverableGatewayError(Exception):
    """Provider、Engine 或 Resolver adapter 的预期能力失败。"""


class DesignatedTopicUnavailableError(WorkspaceDomainError):
    """用户指定的话题已不在这个 Workspace 的候选话题中（已结算或被驱逐）。

    这是对用户选择的拒绝，不是话题路由失败：不进入续接上一轮的兜底，这一轮以本
    错误码失败、不提交，由用户重新选择话题，避免这一轮悄悄落到别的话题上。
    """

    code = "gateway.topic.designated_unavailable"


__all__ = ["DesignatedTopicUnavailableError", "RecoverableGatewayError"]
