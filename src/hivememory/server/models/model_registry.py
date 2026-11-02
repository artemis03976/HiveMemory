"""Model Registry 请求/响应模型"""

from typing import Any

from pydantic import BaseModel, Field

from hivememory.core.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from hivememory.core.models.model_definition import ModelDefinition


class ModelResponse(BaseModel):
    """模型定义的 API 响应体"""

    id: str
    display_name: str
    litellm_model: str
    provider: str
    temperature: float
    max_tokens: int
    context_window: int | None = None
    top_p: float
    is_default: bool

    @classmethod
    def from_definition(cls, model: ModelDefinition) -> "ModelResponse":
        return cls(
            id=model.id,
            display_name=model.display_name,
            litellm_model=model.litellm_model,
            provider=model.provider,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            context_window=model.context_window,
            top_p=model.top_p,
            is_default=model.is_default,
        )


class ModelCreateRequest(BaseModel):
    """创建新模型的请求体"""

    id: str = Field(description="全局唯一标识符，如 'gpt-4o'")
    display_name: str = Field(description="前端展示名称，如 'GPT-4o'")
    litellm_model: str = Field(description="litellm 模型标识符，如 'gpt-4o'")
    provider: str = Field(default="", description="提供商标识，留空自动从 litellm_model 前缀推导")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    context_window: int | None = Field(default=None, gt=0, description="模型总上下文窗口 token 数")
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    is_default: bool = Field(default=False, description="设为系统默认模型")


class ModelUpdateRequest(BaseModel):
    """更新模型的请求体 — 所有字段均为可选，只发送需要修改的字段。"""

    display_name: str | None = None
    litellm_model: str | None = None
    provider: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    context_window: int | None = Field(
        default=None, gt=0, description="总窗口；显式 null 恢复默认预算"
    )
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    is_default: bool | None = None

    def to_updates_dict(self) -> dict[str, Any]:
        """
        只返回客户端实际传入（非 None）的字段，
        防止把未设置的字段误覆盖为 None。

        context_window 显式传 null 时清除登记值，使 Alice 使用配置的默认窗口。
        """
        result: dict[str, Any] = self.model_dump(exclude_none=True)
        if "context_window" in self.model_fields_set:
            result["context_window"] = self.context_window
        return result
