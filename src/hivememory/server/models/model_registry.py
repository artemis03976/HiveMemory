"""Model Registry 请求/响应模型"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from hivememory.core.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from hivememory.core.models.model_definition import ModelDefinition


def _mask_api_key(api_key: Optional[str]) -> Optional[str]:
    """
    对 API 密钥做脱敏处理，避免通过 API 泄露明文密钥。

    规则：
    - None → None（未设置）
    - 长度 ≤ 8 → "***"
    - 长度 > 8 → 保留前 3 位 + "..." + 后 4 位，如 "sk-...abcd"
    """
    if api_key is None:
        return None
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:3]}...{api_key[-4:]}"


class ModelResponse(BaseModel):
    """模型定义的 API 响应体（api_key 已脱敏）"""

    id: str
    display_name: str
    litellm_model: str
    provider: str
    api_key_masked: Optional[str] = Field(
        default=None,
        description="脱敏后的 API 密钥，如 'sk-...abcd'；未设置则为 null"
    )
    api_base: Optional[str] = None
    temperature: float
    max_tokens: int
    top_p: float
    is_default: bool

    @classmethod
    def from_definition(cls, model: ModelDefinition) -> "ModelResponse":
        return cls(
            id=model.id,
            display_name=model.display_name,
            litellm_model=model.litellm_model,
            provider=model.provider,
            api_key_masked=_mask_api_key(model.api_key),
            api_base=model.api_base,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
            is_default=model.is_default,
        )


class ModelCreateRequest(BaseModel):
    """创建新模型的请求体"""

    id: str = Field(description="全局唯一标识符，如 'gpt-4o'")
    display_name: str = Field(description="前端展示名称，如 'GPT-4o'")
    litellm_model: str = Field(description="litellm 模型标识符，如 'gpt-4o'")
    provider: str = Field(default="", description="提供商标识，留空自动从 litellm_model 前缀推导")
    api_key: Optional[str] = Field(default=None, description="API 密钥，留空则由 provider 凭证或环境变量提供")
    api_base: Optional[str] = Field(default=None, description="自定义 API 地址，留空使用默认")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    is_default: bool = Field(default=False, description="设为系统默认模型")


class ModelUpdateRequest(BaseModel):
    """
    更新模型的请求体 — 所有字段均为可选，只发送需要修改的字段。

    注意：api_key 传空字符串 "" 可清除已设置的密钥（改为从环境变量读取）。
    """

    display_name: Optional[str] = None
    litellm_model: Optional[str] = None
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_default: Optional[bool] = None

    def to_updates_dict(self) -> Dict[str, Any]:
        """
        只返回客户端实际传入（非 None）的字段，
        防止把未设置的字段误覆盖为 None。

        特殊处理：api_key="" 表示"清除密钥"，转换为 None 存储。
        """
        result = {}
        for field_name, value in self.model_dump(exclude_none=True).items():
            if field_name == "api_key" and value == "":
                result[field_name] = None
            else:
                result[field_name] = value
        return result
