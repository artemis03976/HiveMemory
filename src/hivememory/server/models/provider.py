"""Provider 请求/响应模型"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ProviderResponse(BaseModel):
    """提供商凭证的 API 响应体（api_key 已脱敏）"""

    name: str
    api_key_masked: Optional[str] = Field(
        default=None,
        description="脱敏后的 API 密钥；未设置则为 null",
    )
    api_base: Optional[str] = None
    is_from_env: bool = Field(
        default=False,
        description="True 表示来自环境变量（只读，不可通过 API 删除/覆盖）",
    )


class ProviderUpsertRequest(BaseModel):
    """创建或更新提供商凭证的请求体"""

    api_key: Optional[str] = Field(
        default=None,
        description="API 密钥。传 null 或不传表示不更改已有值（仅更新 api_base 时使用）",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="自定义 API 地址，留空使用提供商默认值",
    )
