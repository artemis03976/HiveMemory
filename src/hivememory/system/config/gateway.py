from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuleInterceptorConfig(BaseModel):
    enabled: bool = Field(default=True)
    enable_system: bool = Field(default=True)
    enable_chat: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class LLMAnalyzerConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class SystemCommandConfig(BaseModel):
    """
    System Gateway 系统指令配置。

    builtin 使用 command_id -> enabled 的简单覆盖表；复杂参数 schema 和 UI 自动生成
    留到后续阶段，避免 Phase 2.4 扩大配置面。
    """

    enabled: bool = Field(default=True)
    unknown_command_policy: Literal["reject", "ignore"] = Field(default="reject")
    expose_listing: bool = Field(default=True)
    enable_debug_commands: bool = Field(default=False)
    builtin: dict[str, bool] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class GatewayContextHydrationConfig(BaseModel):
    """Gateway Context Hydration 配置。"""

    enabled: bool = Field(default=True)
    timeout_seconds: float = Field(default=1.0, ge=0)
    include_empty_topics: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore")


class SystemGatewayConfig(BaseModel):
    interceptor: RuleInterceptorConfig = Field(default_factory=RuleInterceptorConfig)
    analyzer: LLMAnalyzerConfig = Field(default_factory=LLMAnalyzerConfig)
    commands: SystemCommandConfig = Field(default_factory=SystemCommandConfig)
    context_hydration: GatewayContextHydrationConfig = Field(
        default_factory=GatewayContextHydrationConfig
    )

    model_config = ConfigDict(extra="ignore")


__all__ = [
    "RuleInterceptorConfig",
    "LLMAnalyzerConfig",
    "SystemCommandConfig",
    "GatewayContextHydrationConfig",
    "SystemGatewayConfig",
]
