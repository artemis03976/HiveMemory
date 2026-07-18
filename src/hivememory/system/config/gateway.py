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

    timeout_seconds: float = Field(default=1.0, ge=0)
    include_empty_topics: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore")


class GatewayWorkflowConfig(BaseModel):
    """Gateway workflow 的请求级控制配置。"""

    default_request_timeout_ms: int = Field(default=8000, ge=1)

    model_config = ConfigDict(extra="forbid")


class GatewayContextPreparationConfig(BaseModel):
    """Gateway 两阶段上下文准备配置。"""

    candidate_topics_timeout_ms: int = Field(default=1000, ge=1)
    routed_topic_timeout_ms: int = Field(default=1000, ge=1)
    include_empty_topics: bool = False

    model_config = ConfigDict(extra="forbid")


class TopicRouterConfig(BaseModel):
    """独立话题路由 Engine 配置。"""

    enabled: bool = True
    timeout_ms: int = Field(default=3000, ge=1)
    model_override: str | None = None

    model_config = ConfigDict(extra="forbid")


class UserQueryAnalysisConfig(BaseModel):
    """User Query Analysis 整体 deadline 与保守默认值。"""

    overall_timeout_ms: int = Field(default=5000, ge=1)
    default_mode: Literal["dense", "sparse", "hybrid", "skip"] = "hybrid"
    default_top_k: int = Field(default=5, ge=0)

    model_config = ConfigDict(extra="forbid")


class SystemGatewayConfig(BaseModel):
    interceptor: RuleInterceptorConfig = Field(default_factory=RuleInterceptorConfig)
    analyzer: LLMAnalyzerConfig = Field(default_factory=LLMAnalyzerConfig)
    commands: SystemCommandConfig = Field(default_factory=SystemCommandConfig)
    context_hydration: GatewayContextHydrationConfig = Field(
        default_factory=GatewayContextHydrationConfig
    )
    workflow: GatewayWorkflowConfig = Field(default_factory=GatewayWorkflowConfig)
    context_preparation: GatewayContextPreparationConfig = Field(
        default_factory=GatewayContextPreparationConfig
    )
    topic_router: TopicRouterConfig = Field(default_factory=TopicRouterConfig)
    user_query_analysis: UserQueryAnalysisConfig = Field(
        default_factory=UserQueryAnalysisConfig
    )

    model_config = ConfigDict(extra="ignore")


__all__ = [
    "RuleInterceptorConfig",
    "LLMAnalyzerConfig",
    "SystemCommandConfig",
    "GatewayContextHydrationConfig",
    "GatewayContextPreparationConfig",
    "GatewayWorkflowConfig",
    "TopicRouterConfig",
    "UserQueryAnalysisConfig",
    "SystemGatewayConfig",
]
