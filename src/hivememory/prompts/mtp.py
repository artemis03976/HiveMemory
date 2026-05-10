"""
MTP System Prompt 构建器

生成 MTP 协议的 System Prompt 片段，教导 Worker Agent 使用 MTP 语法。
该片段仅包含协议教学内容，不包含角色设定（persona）或工作区状态。

模块结构:
1. 协议规格 (Protocol Specification, 含内核上下文 header)
2. 负面约束 (Negative Constraints)
3. 行为准则 (Behavioral Guidelines)
4. 内核工具列表 (Kernel Tools, optional)
5. 高密度演示 (Dense Demo, optional)
6. 错误恢复 (Error Recovery, optional)

对应设计文档: MemoryToolProtocol.md Chapter 5

作者: HiveMemory Team
版本: 2.1
"""

import logging
from typing import List, Optional, Tuple

from hivememory.patchouli.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
)

logger = logging.getLogger(__name__)


# ========== MVP 默认内核工具列表 (Chapter 8.6) ==========

DEFAULT_KERNEL_TOOLS: List[Tuple[str, str]] = [
    ("sys_clock", "Get current date, time, and timezone."),
    ("sys_web_search", "Search the internet for latest information."),
    ("sys_read_file", "Read a file from the workspace."),
    ("sys_write_file", "Write content to a file in the workspace."),
    ("sys_python_repl", "Execute Python code for calculation or data processing."),
]


# ========== MTP 动词定义注册表 ==========

_VERB_DEFS_EN = {
    "SEARCH": 'Discover unknown memories. Target=`*`. Args: `query="..."`, optional `filter="type:CODE"` (types: CODE, FACT, URL, REFLECTION, PROFILE, WIP).',
    "READ": "Fetch full content. Target=`alias` or `[alias1, alias2]` (use LIST for batching).",
    "RUN": 'Execute a kernel tool. Target=`tool_alias`. Args: `key="value"`.',
    "WRITE": "Save valuable insights. Target=`*`. Args: `title=\"...\" content=`...``.",
    "UPDATE": "Patch existing memory. Target=`alias`. Args: `patch=`...``.",
    "CALL": 'Delegate to a sub-agent. Target=`agent_alias` (from Available Sub-Agents list). Args: `topic="..."`, optional `context_refs="[alias1, alias2]"` to share memories.',
}

_VERB_DEFS_ZH = {
    "SEARCH": '发现未知记忆。Target=`*`。参数: `query="..."`，可选 `filter="type:CODE"` (类型: CODE, FACT, URL, REFLECTION, PROFILE, WIP)。',
    "READ": "获取完整内容。Target=`alias` 或 `[alias1, alias2]` (使用列表批量读取)。",
    "RUN": '执行内核工具。Target=`tool_alias`。参数: `key="value"`。',
    "WRITE": "保存有价值的洞察。Target=`*`。参数: `title=\"...\" content=`...``。",
    "UPDATE": "修正已有记忆。Target=`alias`。参数: `patch=`...``。",
    "CALL": '委托子代理执行专项任务。Target=`agent_alias` (来自可用子代理列表)。参数: `topic="..."`，可选 `context_refs="[alias1, alias2]"` 共享记忆。',
}

# 默认全量动词顺序
_VERB_ORDER = ["SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"]


# ========== 英文模板 ==========

_PROTOCOL_SPEC_EN = """\
### HIVE MEMORY KERNEL CONTEXT ###

You are an intelligent Agent running on HiveOS. You have access to a persistent memory kernel via the Memory Tool Protocol (MTP).

[PROTOCOL RULES]
1. INTERACTION: Do NOT use JSON or Function Calling. Use MTP syntax directly in your text flow.
2. SYNTAX: `{left_delim} VERB | TARGET | ARGS {right_delim}`
   - Delimiters: `{left_delim}` (open) and `{right_delim}` (close).
   - Separator: `|` (pipe).
3. VERBS:
{verb_list}
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
- Inline Flow: Execute protocol commands as part of your thought process. Do not stop to ask for permission.
- Delegate to Sub-Agents: When the memory context lists Available Sub-Agents and the task matches their specialty, issue CALL to delegate instead of handling it yourself. Pass relevant memory aliases via `context_refs` to share context."""

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
When you receive `<mtp_response status="error">`, check the error category tag:
- [Syntax Error]: You made a protocol mistake. Fix your command syntax and retry.
- [Invalid Argument]: A required argument is missing or malformed. Fix and retry.
- [Alias Not Found]: The alias doesn't exist. Use SEARCH to discover the correct alias, then retry.
- [Memory Not Found]: The memory was archived or deleted. Use SEARCH to find alternatives.
- [Type Mismatch]: The memory type doesn't match the operation. Check the type and use the correct command.
- [Storage Offline]: Memory storage is unavailable. Do NOT retry. Continue without memory.
- [Storage Error]: An internal storage error occurred. Do NOT retry. Continue without memory.
- [Tool Error]: A tool encountered an internal error. Do NOT retry with the same input.
- [Service Unavailable]: A required service is down. Do NOT retry. Continue without memory.
- [Internal Error]: An unexpected error occurred. Do NOT retry. Continue normally.

Rule: If the category says "Do NOT retry", you MUST stop issuing MTP commands and answer from your own knowledge."""


# ========== 中文模板 ==========

_PROTOCOL_SPEC_ZH = """\
### HIVE MEMORY 内核上下文 ###

你是运行在 HiveOS 上的智能 Agent。你可以通过 Memory Tool Protocol (MTP) 访问持久化记忆内核。

[协议规则]
1. 交互方式: 不要使用 JSON 或 Function Calling。直接在文本流中使用 MTP 语法。
2. 语法: `{left_delim} VERB | TARGET | ARGS {right_delim}`
   - 定界符: `{left_delim}` (开) 和 `{right_delim}` (闭)。
   - 分隔符: `|` (管道符)。
3. 指令集:
{verb_list}
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
- 行内执行: 将协议指令作为思考过程的一部分执行，不要停下来请求许可。
- 优先委托: 若记忆上下文中列出了可用子代理，且任务契合其专项能力，应优先使用 CALL 委托子代理执行，而非自行承担。可通过 `context_refs` 传递相关记忆别名以共享上下文。"""

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
当你收到 `<mtp_response status="error">` 时，请检查错误类别标签：
- [Syntax Error]: 你的协议语法有误。修正指令语法后重试。
- [Invalid Argument]: 必需参数缺失或格式错误。修正后重试。
- [Alias Not Found]: 别名不存在。先使用 SEARCH 发现正确的别名，再重试。
- [Memory Not Found]: 记忆已归档或删除。使用 SEARCH 查找替代项。
- [Type Mismatch]: 记忆类型与操作不匹配。检查类型并使用正确的指令。
- [Storage Offline]: 记忆存储不可用。禁止重试。不使用记忆继续对话。
- [Storage Error]: 存储内部错误。禁止重试。不使用记忆继续对话。
- [Tool Error]: 工具遇到内部错误。禁止使用相同输入重试。
- [Service Unavailable]: 所需服务已下线。禁止重试。不使用记忆继续对话。
- [Internal Error]: 发生意外错误。禁止重试。正常继续对话。

规则：如果错误类别包含"禁止重试"，你必须停止发出 MTP 指令，使用自身知识回答用户。"""


# ========== 语言无关模板 ==========

_KERNEL_TOOLS_TEMPLATE = """\
[KERNEL TOOLS] (Available via RUN)
{tool_list}"""


# ========== Prompt 构建器 ==========

class MTPPromptBuilder:
    """
    MTP 协议 System Prompt 构建器

    仅负责生成 MTP 协议教学片段。不包含角色设定（persona）、
    预检索记忆或话题状态 — 这些由上层 SystemPromptBuilder 编排。

    使用示例:
        >>> builder = MTPPromptBuilder(language="en")
        >>> mtp_fragment = builder.build()
    """

    def __init__(
        self,
        language: str = "zh",
        kernel_tools: Optional[List[Tuple[str, str]]] = None,
        include_demo: bool = True,
        include_error_handling: bool = True,
        allowed_verbs: Optional[List[str]] = None,
        allowed_kernel_tools: Optional[List[str]] = None,
    ):
        """
        Args:
            language: 语言 ("zh" 或 "en")
            kernel_tools: 内核工具注册表 [(alias, description), ...]
                         如果为 None，使用 DEFAULT_KERNEL_TOOLS
            include_demo: 是否包含 One-Shot 演示
            include_error_handling: 是否包含错误恢复指令
            allowed_verbs: MTP 动词白名单，None=全量渲染
            allowed_kernel_tools: 系统工具白名单，None=全量渲染，
                                  空列表=不渲染工具列表
        """
        self.language = language
        self.include_demo = include_demo
        self.include_error_handling = include_error_handling

        # 权限过滤：根据白名单过滤工具列表
        base_tools = kernel_tools if kernel_tools is not None else DEFAULT_KERNEL_TOOLS
        if allowed_kernel_tools is not None:
            allowed_set = set(allowed_kernel_tools)
            self.kernel_tools = [(a, d) for a, d in base_tools if a in allowed_set]
        else:
            self.kernel_tools = base_tools

        # 权限过滤：MTP 动词白名单 (用于协议规格渲染)
        self.allowed_verbs = set(v.upper() for v in allowed_verbs) if allowed_verbs else None

    def build(self) -> str:
        """
        构建 MTP 协议教学 System Prompt 片段

        Returns:
            str: MTP 协议教学文本
        """
        sections = []

        # 1. 协议规格 (含内核上下文 header)
        sections.append(self._build_protocol_spec())

        # 2. 负面约束
        sections.append(self._build_negative_constraints())

        # 3. 行为准则
        sections.append(self._build_behavioral_guidelines())

        # 4. 内核工具列表 (有工具时渲染)
        if self.kernel_tools:
            sections.append(self._build_kernel_tools())

        # 5. 高密度演示 (可选)
        if self.include_demo:
            sections.append(self._build_dense_demo())

        # 6. 错误恢复 (可选)
        if self.include_error_handling:
            sections.append(self._build_error_handling())

        return "\n\n".join(sections)

    def _build_protocol_spec(self) -> str:
        """构建协议规格模块，根据 allowed_verbs 动态渲染指令集"""
        template = _PROTOCOL_SPEC_ZH if self.language == "zh" else _PROTOCOL_SPEC_EN
        verb_defs = _VERB_DEFS_ZH if self.language == "zh" else _VERB_DEFS_EN

        verbs = [v for v in _VERB_ORDER if self.allowed_verbs is None or v in self.allowed_verbs]
        verb_lines = [f"   - {v}: {verb_defs[v]}" for v in verbs]

        return template.format(
            left_delim=MTP_LEFT_DELIMITER,
            right_delim=MTP_RIGHT_DELIMITER,
            verb_list="\n".join(verb_lines),
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
    language: str = "zh",
    kernel_tools: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    便捷函数: 获取 MTP System Prompt 片段

    Args:
        language: 语言 ("zh" 或 "en")
        kernel_tools: 可用的内核工具列表

    Returns:
        str: MTP System Prompt 片段
    """
    builder = MTPPromptBuilder(
        language=language,
        kernel_tools=kernel_tools,
    )
    return builder.build()


__all__ = [
    "MTPPromptBuilder",
    "get_mtp_prompt",
    "DEFAULT_KERNEL_TOOLS",
]
