from pydantic import BaseModel, Field, ConfigDict


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


class AliceConfig(BaseModel):
    koakuma: KoakumaConfig = Field(default_factory=KoakumaConfig)
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)

    model_config = ConfigDict(extra="ignore")
