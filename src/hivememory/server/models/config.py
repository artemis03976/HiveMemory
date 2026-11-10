"""Config 请求/响应模型"""

from typing import Any

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """主配置的读取与保存结果。"""

    config: dict[str, Any] = Field(
        description="期望配置：下次启动将装载的完整配置树，密钥只返回打码值"
    )
    immediate_fields: list[str] = Field(
        description="保存后立即生效的字段路径（以点分隔），其余字段在重启后生效"
    )
    pending_restart: list[str] = Field(
        description="期望值与运行值不同、等待重启的字段路径（以点分隔）"
    )
