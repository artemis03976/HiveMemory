"""
HiveMemory 模型注册表 — 数据模型

ModelDefinition 是注册表中单条模型记录的数据结构；ModelSelection 是使用方对
注册表模型的选择，以及对生成参数的覆盖。AgentProfile.model_name 与
ModelSelection.model_id 通过 id 字段引用对应的模型定义。
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


class ModelDefinition(BaseModel):
    """
    模型定义 — 注册表中的单条记录

    每条记录描述一个可用的 LLM，包含前端展示所需的显示信息
    以及 litellm 调用所需的技术参数。

    ModelRegistry.resolve() 把使用方的 ModelSelection 与选中记录的默认值合并为 LLM 调用的
    请求参数；上下文窗口与展示名不属于请求参数，需要时经 ModelRegistry.select_model()
    取得记录本身。

    凭证解析（见 ModelRegistry.resolve）：
    - 密钥与 API 地址只来自 provider 字段对应的 Provider 凭证；models.yaml 受版本控制，
      模型定义不保存连接凭证。单个模型需要不同的密钥或地址时，另登记一个 provider
    """

    id: str = Field(
        description="全局唯一标识符，如 'deepseek-chat'。AgentProfile.model_name 通过此 ID 引用模型"
    )
    display_name: str = Field(description="前端展示名称，如 'DeepSeek Chat'")
    litellm_model: str = Field(
        description=(
            "传递给 litellm.completion() 的完整模型标识符，"
            "格式为 'provider/model-name'，如 'deepseek/deepseek-chat'、'gpt-4o'"
        )
    )
    provider: str = Field(
        default="",
        description=(
            "提供商标识，用于查 Provider 凭证，如 'deepseek'、'openai'。"
            "留空时自动从 litellm_model 的前缀推导（'deepseek/xxx' → 'deepseek'）"
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="默认推理温度，Agent Profile 或会话请求可以覆盖此值",
    )
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0, description="默认最大生成 token 数")
    context_window: int | None = Field(default=None, gt=0, description="模型总上下文窗口 token 数")
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="默认核采样阈值，Agent Profile 或会话请求可以覆盖此值",
    )
    is_default: bool = Field(
        default=False,
        description=(
            "是否为系统默认模型。当 AgentProfile.model_name='default' 时，"
            "注册表将使用此模型。注册表中有且仅有一条记录的 is_default 应为 True"
        ),
    )

    @model_validator(mode="after")
    def _derive_provider(self) -> "ModelDefinition":
        """provider 留空时从 litellm_model 前缀推导（'deepseek/xxx' → 'deepseek'）。"""
        if not self.provider and self.litellm_model and "/" in self.litellm_model:
            self.provider = self.litellm_model.split("/", 1)[0]
        return self


class ModelSelection(BaseModel):
    """使用方对注册表模型的选择，以及按使用方需要对生成参数的覆盖。

    每一项为 None 时表示这一层不覆盖、沿用上一层：会话请求的选择沿用 Agent Profile，
    Profile 与组件配置（Gateway、Patchouli）沿用模型定义；``model_id`` 一路都没有给出时
    取注册表的默认模型。生成参数跟着使用方而不是模型定义，因为它们常常是使用方任务的
    要求（例如一次结算输出多条记忆所需的输出上限），放进模型定义会随模型记录被其他
    使用方连带修改。
    """

    model_id: str | None = Field(
        default=None,
        min_length=1,
        description="模型注册表 ID，'default' 表示注册表的默认模型；None 表示沿用上一层",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="采样温度覆盖")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="核采样阈值覆盖")
    max_tokens: int | None = Field(default=None, ge=1, description="最大生成 token 数覆盖")

    model_config = ConfigDict(extra="forbid")
