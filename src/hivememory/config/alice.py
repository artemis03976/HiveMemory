from pydantic import BaseModel, ConfigDict, Field


class MTPPromptConfig(BaseModel):
    enabled: bool = Field(default=True)
    include_demo: bool = Field(default=True)
    include_error_handling: bool = Field(default=True)

    model_config = ConfigDict(extra="ignore")


class KoakumaConfig(BaseModel):
    enabled: bool = Field(default=True)
    execution_timeout_seconds: int = Field(default=30)
    tool_cache_size: int = Field(default=64)
    python_repl_timeout_seconds: int = Field(default=10)
    workspace_path: str = Field(default="./workspace")
    file_read_max_bytes: int = Field(default=102400)
    file_write_max_bytes: int = Field(default=102400)
    web_search_timeout_seconds: int = Field(default=15)
    mtp_prompt: MTPPromptConfig = Field(default_factory=MTPPromptConfig)

    model_config = ConfigDict(extra="ignore")


class AgentRuntimeConfig(BaseModel):
    max_loop_iterations: int = Field(default=10)

    model_config = ConfigDict(extra="ignore")


class CompactionConfig(BaseModel):
    """会话上下文预算；未知模型窗口时使用保守默认值。"""

    default_context_window: int = Field(default=65536, gt=0, description="默认总窗口 token 数")
    safety_margin: int = Field(default=2048, ge=0, description="为估算误差预留的 token 数")

    model_config = ConfigDict(extra="ignore")


class AliceConfig(BaseModel):
    koakuma: KoakumaConfig = Field(default_factory=KoakumaConfig)
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)

    model_config = ConfigDict(extra="ignore")
