"""Import Bus 配置段。

Import Bus 属于 System 的被动摄入能力，不并入 Gateway 配置；Gateway 只
识别固定的 ``GatewayIngressMode``，请求时限与最近轮数沿用 Gateway 自己的
配置段。
"""

from pydantic import BaseModel, ConfigDict, Field


class ImportBusConfig(BaseModel):
    """导入一轮完整交互记录的入口约束。"""

    max_request_bytes: int = Field(
        default=8 * 1024 * 1024,
        gt=0,
        description="单次导入请求体的字节上限，超出以 413 拒绝且不可重试",
    )

    model_config = ConfigDict(extra="forbid")


__all__ = ["ImportBusConfig"]
