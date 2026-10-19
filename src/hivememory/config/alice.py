from pydantic import BaseModel, ConfigDict, Field


class MTPPromptConfig(BaseModel):
    enabled: bool = Field(default=True)
    include_demo: bool = Field(default=True)
    include_error_handling: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class KoakumaConfig(BaseModel):
    enabled: bool = Field(default=True)
    python_repl_timeout_seconds: int = Field(default=10)
    workspace_path: str = Field(default="./workspace")
    file_read_max_bytes: int = Field(default=102400)
    file_write_max_bytes: int = Field(default=102400)
    web_search_timeout_seconds: int = Field(default=15)
    mtp_prompt: MTPPromptConfig = Field(default_factory=MTPPromptConfig)

    model_config = ConfigDict(extra="forbid")


class AgentRuntimeConfig(BaseModel):
    """单帧循环上限与模型输入预算；未知模型窗口时使用保守默认值。"""

    default_context_window: int = Field(default=65536, gt=0, description="默认总窗口 token 数")
    safety_margin: int = Field(default=2048, ge=0, description="为估算误差预留的 token 数")
    max_loop_iterations: int = Field(default=10)

    model_config = ConfigDict(extra="forbid")


class CompactionConfig(BaseModel):
    """历史压缩的触发阈值、保留比例与摘要生成参数。"""

    enabled: bool = Field(default=True, description="启用历史压缩，不影响轮内预算守卫")
    trigger_ratio: float = Field(default=0.75, gt=0, le=1, description="触发压缩的输入预算比例")
    recent_raw_ratio: float = Field(
        default=0.25, ge=0, le=1, description="压缩后近期原文可占用的输入预算比例"
    )
    tool_result_excerpt_chars: int = Field(
        default=2000, gt=0, description="RUN 摘录及保留型工具结果的最大正文字符数"
    )
    target_ratio: float = Field(default=0.5, gt=0, le=1, description="压缩后的目标输入预算比例")
    summary_max_tokens: int = Field(default=1024, gt=0, description="接力摘要正文的 token 上限")
    summary_input_tool_chars: int = Field(
        default=2000, gt=0, description="摘要请求中每条工具结果的最大字符数"
    )
    summary_timeout_seconds: float = Field(default=60, gt=0, description="每次摘要调用的超时秒数")
    max_references: int = Field(default=16, ge=0, description="摘要引用线索的最大条数")
    reference_max_chars: int = Field(default=256, gt=0, description="单条引用线索的最大字符数")

    model_config = ConfigDict(extra="forbid")


class AliceConfig(BaseModel):
    koakuma: KoakumaConfig = Field(default_factory=KoakumaConfig)
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)

    model_config = ConfigDict(extra="forbid")
