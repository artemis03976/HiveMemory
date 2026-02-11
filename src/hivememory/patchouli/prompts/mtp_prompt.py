"""
MTP System Prompt 构建器

生成 MTP 协议的 System Prompt 片段，教导 Worker Agent 使用 MTP 语法。

四大模块 (Section 5.1.1):
1. 角色定义 (Role Definition)
2. 协议规格 (Protocol Specification)
3. 负面约束 (Negative Constraints)
4. 高密度演示 (Dense Demo)

附加模块:
5. 错误恢复 (Error Handling, Section 5.3)
6. 内核工具列表 (Kernel Tools, Chapter 8.7)

对应设计文档: MemoryToolProtocol.md Chapter 5 & 8.7

作者: HiveMemory Team
版本: 1.0
"""

import logging
from enum import Enum
from typing import List, Optional, Tuple

from hivememory.patchouli.protocol.mtp import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
)

logger = logging.getLogger(__name__)


# ========== Agent 角色枚举 ==========

class AgentRole(str, Enum):
    """
    Worker Agent 角色类型 (Section 5.2.2)

    控制 MTP 触发倾向:
    - CODER: 激进查阅，必须先查阅已有代码库再编写新代码
    - CHAT: 保守查阅，仅在必要时查阅记忆
    - DEFAULT: 平衡模式
    """
    CODER = "coder"
    CHAT = "chat"
    DEFAULT = "default"


# ========== MVP 默认内核工具列表 (Chapter 8.6) ==========

DEFAULT_KERNEL_TOOLS: List[Tuple[str, str]] = [
    ("sys_clock", "Get current date, time, and timezone."),
    ("sys_web_search", "Search the internet for latest information."),
    ("sys_read_file", "Read a file from the workspace."),
    ("sys_write_file", "Write content to a file in the workspace."),
    ("sys_python_repl", "Execute Python code for calculation or data processing."),
]


# ========== 英文模板 ==========

_ROLE_DEFINITION_EN = """\
### HIVE MEMORY KERNEL CONTEXT ###

You are an intelligent Agent running on HiveOS. You have access to a persistent memory kernel via the Memory Tool Protocol (MTP).
{role_instruction}"""

_ROLE_INSTRUCTION_CODER_EN = (
    "You are a rigorous engineer. You MUST consult existing code and documentation "
    "in HiveMemory before writing new code. Always verify facts via SEARCH/READ."
)
_ROLE_INSTRUCTION_CHAT_EN = (
    "You are a helpful assistant. Consult HiveMemory only when necessary "
    "to answer factual questions or retrieve specific information."
)
_ROLE_INSTRUCTION_DEFAULT_EN = (
    "When you lack context to answer truthfully, consult HiveMemory. "
    "Do not guess about facts, code, or configurations that may exist in memory."
)

_PROTOCOL_SPEC_EN = """\
[PROTOCOL RULES]
1. INTERACTION: Do NOT use JSON or Function Calling. Use MTP syntax directly in your text flow.
2. SYNTAX: `{left_delim} VERB | TARGET | ARGS {right_delim}`
   - Delimiters: `{left_delim}` (open) and `{right_delim}` (close).
   - Separator: `|` (pipe).
3. VERBS:
   - SEARCH: Discover unknown memories. Target=`*`. Args: `query="..."`, optional `filter="type:CODE"` (types: CODE, FACT, URL, REFLECTION, PROFILE, WIP).
   - READ: Fetch full content. Target=`alias` or `[alias1, alias2]` (use LIST for batching).
   - RUN: Execute a kernel tool. Target=`tool_alias`. Args: `key="value"`.
   - WRITE: Save valuable insights. Target=`*`. Args: `title="..." content=`...``.
   - UPDATE: Patch existing memory. Target=`alias`. Args: `patch=`...``.
4. RESPONSE: Results appear in `<mtp_response>` XML blocks immediately after your command."""

_NEGATIVE_CONSTRAINTS_EN = """\
[CONSTRAINTS]
- NEVER output JSON or Function Calling format for memory operations.
- NEVER fabricate memory aliases (e.g., `mem_xyz`) that were not returned by SEARCH. If unsure, SEARCH first.
- NEVER use raw UUIDs. Always use semantic aliases returned by the system."""

_BEHAVIORAL_GUIDELINES_EN = """\
[BEHAVIORAL GUIDELINES]
- Verify First: If asked about specific facts, code, or configurations, SEARCH/READ memory first. Do not guess.
- Batch Operations: Always group multiple READ requests into one list `[a, b, c]` to save IO cycles.
- Inline Flow: Execute protocol commands as part of your thought process. Do not stop to ask for permission."""

_DENSE_DEMO_EN = """\
[ONE-SHOT DEMONSTRATION]
User: "Deploy the login module using the standard config."
Assistant: I need to find the login deployment script and the standard config.
{left_delim} SEARCH | * | query="login deploy script" {right_delim}
<mtp_response status="success">
[Menu]:
1. tool_deploy_login (Alias) - "Python script to deploy login module via SSH."
2. fact_std_config (Alias) - "Standard deployment configuration variables."
</mtp_response>
Found them. Let me read both to ensure compatibility.
{left_delim} READ | [tool_deploy_login, fact_std_config] | {right_delim}
<mtp_response status="success">
[tool_deploy_login]: def deploy(env, config): ...
[fact_std_config]: ENV_VARS = {{...}}
</mtp_response>
The data is loaded. Now executing the deployment.
{left_delim} RUN | tool_deploy_login | env="prod" config_ref="fact_std_config" {right_delim}
<mtp_response status="success">
Deployment completed successfully. Service is UP.
</mtp_response>
Deployment is complete. The login module is now running in production."""

_ERROR_HANDLING_EN = """\
[ERROR RECOVERY]
If you receive `<mtp_response status="error">`, analyze the error message and retry with the corrected command immediately. Common fixes:
- "Alias not found" -> Use SEARCH to discover the correct alias first.
- "Syntax error" -> Check your delimiter and separator placement.
- "Missing argument" -> Review the required args for the tool and retry."""


# ========== 中文模板 ==========

_ROLE_DEFINITION_ZH = """\
### HIVE MEMORY 内核上下文 ###

你是运行在 HiveOS 上的智能 Agent。你可以通过 Memory Tool Protocol (MTP) 访问持久化记忆内核。
{role_instruction}"""

_ROLE_INSTRUCTION_CODER_ZH = (
    "你是严谨的工程师。在编写新代码之前，你必须先通过 SEARCH/READ 查阅 HiveMemory 中已有的代码和文档。"
    "始终通过记忆验证事实。"
)
_ROLE_INSTRUCTION_CHAT_ZH = (
    "你是得力的助手。仅在需要回答事实性问题或检索特定信息时查阅 HiveMemory。"
)
_ROLE_INSTRUCTION_DEFAULT_ZH = (
    "当你缺乏上下文来如实回答时，请查阅 HiveMemory。"
    "不要猜测可能存在于记忆中的事实、代码或配置。"
)

_PROTOCOL_SPEC_ZH = """\
[协议规则]
1. 交互方式: 不要使用 JSON 或 Function Calling。直接在文本流中使用 MTP 语法。
2. 语法: `{left_delim} VERB | TARGET | ARGS {right_delim}`
   - 定界符: `{left_delim}` (开) 和 `{right_delim}` (闭)。
   - 分隔符: `|` (管道符)。
3. 指令集:
   - SEARCH: 发现未知记忆。Target=`*`。参数: `query="..."`，可选 `filter="type:CODE"` (类型: CODE, FACT, URL, REFLECTION, PROFILE, WIP)。
   - READ: 获取完整内容。Target=`alias` 或 `[alias1, alias2]` (使用列表批量读取)。
   - RUN: 执行内核工具。Target=`tool_alias`。参数: `key="value"`。
   - WRITE: 保存有价值的洞察。Target=`*`。参数: `title="..." content=`...``。
   - UPDATE: 修正已有记忆。Target=`alias`。参数: `patch=`...``。
4. 响应: 执行结果会以 `<mtp_response>` XML 块的形式出现在你的指令之后。"""

_NEGATIVE_CONSTRAINTS_ZH = """\
[约束]
- 绝对不要为记忆操作输出 JSON 或 Function Calling 格式。
- 绝对不要编造未经 SEARCH 返回的记忆别名 (如 `mem_xyz`)。不确定时先 SEARCH。
- 绝对不要使用裸 UUID。始终使用系统返回的语义化别名。"""

_BEHAVIORAL_GUIDELINES_ZH = """\
[行为准则]
- 先验证: 当被问及具体事实、代码或配置时，先 SEARCH/READ 记忆。不要猜测。
- 批量操作: 将多个 READ 请求合并为一个列表 `[a, b, c]`，节省 IO 开销。
- 行内执行: 将协议指令作为思考过程的一部分执行，不要停下来请求许可。"""

_DENSE_DEMO_ZH = """\
[示例演示]
用户: "用标准配置部署登录模块。"
助手: 我需要找到登录部署脚本和标准配置。
{left_delim} SEARCH | * | query="login deploy script" {right_delim}
<mtp_response status="success">
[Menu]:
1. tool_deploy_login (Alias) - "通过 SSH 部署登录模块的 Python 脚本。"
2. fact_std_config (Alias) - "标准部署配置变量。"
</mtp_response>
找到了。我读取两者以确保兼容性。
{left_delim} READ | [tool_deploy_login, fact_std_config] | {right_delim}
<mtp_response status="success">
[tool_deploy_login]: def deploy(env, config): ...
[fact_std_config]: ENV_VARS = {{...}}
</mtp_response>
数据已加载。现在执行部署。
{left_delim} RUN | tool_deploy_login | env="prod" config_ref="fact_std_config" {right_delim}
<mtp_response status="success">
部署成功完成。服务已启动。
</mtp_response>
部署完成。登录模块已在生产环境运行。"""

_ERROR_HANDLING_ZH = """\
[错误恢复]
如果你收到 `<mtp_response status="error">`，请分析错误信息并立即用修正后的指令重试。常见修复:
- "Alias not found" -> 先使用 SEARCH 发现正确的别名。
- "Syntax error" -> 检查定界符和分隔符的位置。
- "Missing argument" -> 查看该工具所需的参数并重试。"""


# ========== 语言无关模板 ==========

_KERNEL_TOOLS_TEMPLATE = """\
[KERNEL TOOLS] (Available via RUN)
{tool_list}"""


# ========== 角色指令映射 ==========

_ROLE_INSTRUCTIONS = {
    "en": {
        AgentRole.CODER: _ROLE_INSTRUCTION_CODER_EN,
        AgentRole.CHAT: _ROLE_INSTRUCTION_CHAT_EN,
        AgentRole.DEFAULT: _ROLE_INSTRUCTION_DEFAULT_EN,
    },
    "zh": {
        AgentRole.CODER: _ROLE_INSTRUCTION_CODER_ZH,
        AgentRole.CHAT: _ROLE_INSTRUCTION_CHAT_ZH,
        AgentRole.DEFAULT: _ROLE_INSTRUCTION_DEFAULT_ZH,
    },
}


# ========== Prompt 构建器 ==========

class MTPPromptBuilder:
    """
    MTP System Prompt 构建器 (Section 5.1)

    组装 MTP 协议的 System Prompt 片段。该片段追加到 Worker Agent
    的基础 System Prompt 之后，教导 LLM 使用 MTP 语法。

    六大模块:
    1. 角色定义 (Role Definition)
    2. 协议规格 (Protocol Specification)
    3. 负面约束 (Negative Constraints)
    4. 行为准则 (Behavioral Guidelines)
    5. 内核工具列表 (Kernel Tools, optional)
    6. 高密度演示 (Dense Demo, optional)
    7. 错误恢复 (Error Handling, optional)

    使用示例:
        >>> builder = MTPPromptBuilder(role=AgentRole.CODER, language="en")
        >>> fragment = builder.build()
        >>> full_prompt = base_prompt + "\\n\\n" + fragment
    """

    def __init__(
        self,
        role: AgentRole = AgentRole.DEFAULT,
        language: str = "zh",
        kernel_tools: Optional[List[Tuple[str, str]]] = None,
        include_demo: bool = True,
        include_error_handling: bool = True,
        include_kernel_tools: bool = True,
    ):
        """
        Args:
            role: Agent 角色类型，控制触发倾向
            language: 语言 ("zh" 或 "en")
            kernel_tools: 可用的内核工具列表 [(alias, description), ...]
                         如果为 None，使用 DEFAULT_KERNEL_TOOLS
            include_demo: 是否包含 One-Shot 演示
            include_error_handling: 是否包含错误恢复指令
            include_kernel_tools: 是否包含内核工具列表
        """
        self.role = role
        self.language = language
        self.kernel_tools = kernel_tools if kernel_tools is not None else DEFAULT_KERNEL_TOOLS
        self.include_demo = include_demo
        self.include_error_handling = include_error_handling
        self.include_kernel_tools = include_kernel_tools

    def build(self) -> str:
        """
        构建完整的 MTP System Prompt 片段

        Returns:
            str: 可直接追加到 Worker Agent System Prompt 的文本片段
        """
        sections = []

        # 1. 角色定义
        sections.append(self._build_role_definition())

        # 2. 协议规格
        sections.append(self._build_protocol_spec())

        # 3. 负面约束
        sections.append(self._build_negative_constraints())

        # 4. 行为准则
        sections.append(self._build_behavioral_guidelines())

        # 5. 内核工具列表 (可选)
        if self.include_kernel_tools and self.kernel_tools:
            sections.append(self._build_kernel_tools())

        # 6. 高密度演示 (可选)
        if self.include_demo:
            sections.append(self._build_dense_demo())

        # 7. 错误恢复 (可选)
        if self.include_error_handling:
            sections.append(self._build_error_handling())

        return "\n\n".join(sections)

    def _build_role_definition(self) -> str:
        """构建角色定义模块"""
        template = _ROLE_DEFINITION_ZH if self.language == "zh" else _ROLE_DEFINITION_EN

        lang_instructions = _ROLE_INSTRUCTIONS.get(self.language, _ROLE_INSTRUCTIONS["en"])
        role_instruction = lang_instructions.get(self.role, lang_instructions[AgentRole.DEFAULT])

        return template.format(role_instruction=role_instruction)

    def _build_protocol_spec(self) -> str:
        """构建协议规格模块"""
        template = _PROTOCOL_SPEC_ZH if self.language == "zh" else _PROTOCOL_SPEC_EN
        return template.format(
            left_delim=MTP_LEFT_DELIMITER,
            right_delim=MTP_RIGHT_DELIMITER,
        )

    def _build_negative_constraints(self) -> str:
        """构建负面约束模块"""
        return _NEGATIVE_CONSTRAINTS_ZH if self.language == "zh" else _NEGATIVE_CONSTRAINTS_EN

    def _build_behavioral_guidelines(self) -> str:
        """构建行为准则模块"""
        return _BEHAVIORAL_GUIDELINES_ZH if self.language == "zh" else _BEHAVIORAL_GUIDELINES_EN

    def _build_kernel_tools(self) -> str:
        """构建内核工具列表"""
        lines = []
        for alias, desc in self.kernel_tools:
            lines.append(f"- `{alias}`: {desc}")
        tool_list = "\n".join(lines)
        return _KERNEL_TOOLS_TEMPLATE.format(tool_list=tool_list)

    def _build_dense_demo(self) -> str:
        """构建高密度演示模块"""
        template = _DENSE_DEMO_ZH if self.language == "zh" else _DENSE_DEMO_EN
        return template.format(
            left_delim=MTP_LEFT_DELIMITER,
            right_delim=MTP_RIGHT_DELIMITER,
        )

    def _build_error_handling(self) -> str:
        """构建错误处理指令"""
        return _ERROR_HANDLING_ZH if self.language == "zh" else _ERROR_HANDLING_EN


# ========== 便捷函数 ==========

def get_mtp_prompt(
    role: AgentRole = AgentRole.DEFAULT,
    language: str = "zh",
    kernel_tools: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    便捷函数: 获取 MTP System Prompt 片段

    Args:
        role: Agent 角色类型
        language: 语言 ("zh" 或 "en")
        kernel_tools: 可用的内核工具列表

    Returns:
        str: MTP System Prompt 片段
    """
    builder = MTPPromptBuilder(
        role=role,
        language=language,
        kernel_tools=kernel_tools,
    )
    return builder.build()


__all__ = [
    "MTPPromptBuilder",
    "get_mtp_prompt",
    "AgentRole",
    "DEFAULT_KERNEL_TOOLS",
]
