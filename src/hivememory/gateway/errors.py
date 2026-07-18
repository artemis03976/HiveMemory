"""Gateway 内部能力适配错误。"""


class RecoverableGatewayError(Exception):
    """Provider、Engine 或 Resolver adapter 的预期能力失败。"""


__all__ = ["RecoverableGatewayError"]
