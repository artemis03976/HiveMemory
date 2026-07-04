from pydantic import BaseModel, ConfigDict, Field


class RuleInterceptorConfig(BaseModel):
    enabled: bool = Field(default=True)
    enable_system: bool = Field(default=True)
    enable_chat: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class LLMAnalyzerConfig(BaseModel):
    enabled: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class SystemGatewayConfig(BaseModel):
    interceptor: RuleInterceptorConfig = Field(default_factory=RuleInterceptorConfig)
    analyzer: LLMAnalyzerConfig = Field(default_factory=LLMAnalyzerConfig)

    model_config = ConfigDict(extra="ignore")


# Backward-compatible name for configs that still refer to the memory-era gateway.
MemoryGatewayConfig = SystemGatewayConfig


__all__ = [
    "RuleInterceptorConfig",
    "LLMAnalyzerConfig",
    "SystemGatewayConfig",
    "MemoryGatewayConfig",
]
