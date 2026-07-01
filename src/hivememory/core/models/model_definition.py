"""
HiveMemory 模型注册表 — 数据模型

ModelDefinition 是注册表中单条模型记录的数据结构。
AgentProfile.model_name 通过 id 字段引用对应的模型定义。
"""

from typing import Optional

from pydantic import BaseModel, Field

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

    与 LLMConfig 的关系：
    - LLMConfig 是实际传给 LiteLLMService 的运行时配置
    - ModelDefinition 是持久化的"模型档案"，可以通过
      ModelRegistry.to_llm_config() 转换为 LLMConfig
    """

    id: str = Field(
        description="全局唯一标识符，如 'deepseek-chat'。AgentProfile.model_name 通过此 ID 引用模型"
    )
    display_name: str = Field(
        description="前端展示名称，如 'DeepSeek Chat'"
    )
    litellm_model: str = Field(
        description=(
            "传递给 litellm.completion() 的完整模型标识符，"
            "格式为 'provider/model-name'，如 'deepseek/deepseek-chat'、'gpt-4o'"
        )
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API 密钥。None 表示由 litellm 从环境变量自动读取"
    )
    api_base: Optional[str] = Field(
        default=None,
        description="自定义 API 基础 URL，用于私有化部署或代理。None 表示使用提供商默认地址"
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="默认推理温度，Agent Profile 或会话请求可以覆盖此值"
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        gt=0,
        description="默认最大生成 token 数"
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="默认核采样阈值，Agent Profile 或会话请求可以覆盖此值"
    )
    is_default: bool = Field(
        default=False,
        description=(
            "是否为系统默认模型。当 AgentProfile.model_name='default' 时，"
            "注册表将使用此模型。注册表中有且仅有一条记录的 is_default 应为 True"
        )
    )