"""
HiveMemory 核心数据模型 - 智能体领域

定义与多智能体系统（Agentic System）相关的数据模型。
"""

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hivememory.core.models.memory import MemoryAtom


class AgentProfile(BaseModel):
    """
    人偶图纸配置 - Agent 运行时的完整配置信息

    身份 (Identity): 由 agent_id 字段承载，来自源原子的 index.alias；
        alias 即 AgentProfile 的 actor 面向身份，由模型自持，内置为 omni_doll
    灵魂 (Persona): 由 persona 字段承载，来自 MemoryAtom.payload.content
    骨架 (Skeleton): 模型参数与语言，来自 MemoryAtom.payload.agent_config

    可见性不进模型：源原子的 ``MemoryAccessPolicy`` 只随 Profile 读取 backing
    结果（``ResolvedAgentProfile``）流动，不属于能力描述。

    Agent 操作权限由 Workspace 访问登记持有，不附着在 Profile 上。
    存量配置中的多余字段按 Pydantic 默认规则忽略，无需迁移。
    """

    agent_id: str | None = Field(
        default=None, description="Agent 身份（源原子 alias）；内置 Profile 为 omni_doll"
    )
    persona: str = Field(default="", description="Agent 人设提示词")
    model_name: str = Field(default="default", description="基底模型名称")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="推理温度覆盖。None 表示沿用注册表模型定义的温度"
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="核采样阈值覆盖。None 表示沿用注册表模型定义的 top_p",
    )

    language: str = Field(default="zh", description="提示词语言 (zh/en)")

    @classmethod
    def from_atom(cls, atom: "MemoryAtom") -> Optional["AgentProfile"]:
        """从 MemoryAtom 解析 AgentProfile（包含身份、persona 和 config）。

        ``agent_id`` 始终取自 ``index.alias``，覆盖 agent_config 中的同名键：
        身份由资源寻址事实决定，不由可编辑配置自报。
        """
        raw = atom.payload.agent_config
        if raw is None:
            return None

        try:
            # 从 payload.agent_config 解析配置，从 payload.content 获取 persona
            config = cls(**{**raw, "persona": atom.payload.content, "agent_id": atom.index.alias})
            return config
        except Exception:
            return None


OMNI_DOLL_PROFILE = AgentProfile(
    agent_id="omni_doll",
    persona="",
    model_name="default",
    temperature=None,
    top_p=None,
    language="zh",
)
"""全能人偶 (Omni-Doll) 默认配置，无特定人设。

temperature/top_p 为 None，运行时沿用注册表默认模型的参数；操作权限由访问登记提供。
"""
