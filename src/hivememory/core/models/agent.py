"""
HiveMemory 核心数据模型 - 智能体领域

定义与多智能体系统（Agentic System）相关的数据模型。
"""

from typing import List, Optional, Set, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hivememory.core.models.memory import MemoryAtom


class AgentProfile(BaseModel):
    """
    人偶图纸配置 - Agent 运行时的完整配置信息

    灵魂 (Persona): 由 persona 字段承载，来自 MemoryAtom.payload.content
    骨架 (Skeleton): 模型参数 + 权限控制表，来自 MemoryAtom.payload.artifacts.agent_config

    权限语义：
    - None = 全部允许（omni_doll 默认行为）
    - [] 空列表 = 禁止所有
    - 非空列表 = 白名单模式，仅允许列表中的项目
    """
    persona: str = Field(default="", description="Agent 人设提示词")
    model_name: str = Field(default="default", description="基底模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="推理温度")

    allowed_mtp_verbs: Optional[List[str]] = Field(
        default=None,
        description="允许的 MTP 指令动词白名单，None=全部允许，[]=禁止所有"
    )
    allowed_sys_tools: Optional[List[str]] = Field(
        default=None,
        description="允许的系统工具白名单，None=全部允许，[]=禁止所有"
    )

    language: str = Field(default="zh", description="提示词语言 (zh/en)")

    _verb_set: Optional[Set[str]] = None
    _tool_set: Optional[Set[str]] = None

    @classmethod
    def from_atom(cls, atom: "MemoryAtom") -> Optional["AgentProfile"]:
        """从 MemoryAtom 解析 AgentProfile（包含 persona 和 config）。"""
        raw = atom.payload.artifacts.agent_config
        if raw is None:
            return None

        try:
            # 从 artifacts.agent_config 解析配置，从 payload.content 获取 persona
            config = cls(
                persona=atom.payload.content,
                **raw
            )
            return config
        except Exception:
            return None

    def get_verb_set(self) -> Set[str]:
        """获取 MTP 动词白名单的 set 版本（惰性构建）"""
        if self._verb_set is None:
            self._verb_set = set(v.upper() for v in self.allowed_mtp_verbs) if self.allowed_mtp_verbs else set()
        return self._verb_set

    def get_tool_set(self) -> Set[str]:
        """获取系统工具白名单的 set 版本（惰性构建）"""
        if self._tool_set is None:
            self._tool_set = set(self.allowed_sys_tools) if self.allowed_sys_tools else set()
        return self._tool_set

    def is_verb_allowed(self, verb: str) -> bool:
        """检查 MTP 动词是否被允许

        权限判断：
        - None: 全部允许
        - []: 禁止所有
        - ["X", "Y"]: 仅允许白名单中的项目
        """
        if self.allowed_mtp_verbs is None:
            return True
        if len(self.allowed_mtp_verbs) == 0:
            return False
        return verb.upper() in self.get_verb_set()

    def is_tool_allowed(self, tool_alias: str) -> bool:
        """检查系统工具是否被允许

        权限判断：
        - None: 全部允许
        - []: 禁止所有
        - ["X", "Y"]: 仅允许白名单中的项目
        """
        if self.allowed_sys_tools is None:
            return True
        if len(self.allowed_sys_tools) == 0:
            return False
        return tool_alias in self.get_tool_set()


OMNI_DOLL_PROFILE = AgentProfile(
    persona="",
    model_name="default",
    temperature=0.7,
    allowed_mtp_verbs=None,
    allowed_sys_tools=None,
    language="zh",
)
"""全能人偶 (Omni-Doll) 默认配置 - 拥有完整权限，无特定人设"""
