"""
MTP System Prompt 构建器

生成 MTP 协议的 System Prompt 片段，教导 Worker Agent 使用 MTP 语法。
该片段仅包含协议教学内容，不包含角色设定（persona）或工作区状态。

模块结构:
1. 协议规格 (Protocol Specification, 含运行时上下文 header)
2. 负面约束 (Negative Constraints)
3. 行为准则 (Behavioral Guidelines)
4. 运行时工具列表 (Runtime Tools, optional)
5. 高密度演示 (Dense Demo, optional)
6. 错误恢复 (Error Recovery, optional)

对应设计文档: MemoryToolProtocol.md Chapter 5

作者: HiveMemory Team
版本: 2.1
"""

import logging
from typing import List, Optional, Tuple

from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
)
from hivememory.i18n import get_mtp_prompt_text, get_mtp_verb_text, resolve_language

logger = logging.getLogger(__name__)


# ========== MVP 默认运行时工具列表 (Chapter 8.6) ==========

DEFAULT_RUNTIME_TOOLS: List[Tuple[str, str]] = [
    ("sys_clock", "Get current date, time, and timezone."),
    ("sys_web_search", "Search the internet for latest information."),
    ("sys_read_file", "Read a file from the workspace."),
    ("sys_write_file", "Write content to a file in the workspace."),
    ("sys_python_repl", "Execute Python code for calculation or data processing."),
]


# 默认全量动词顺序
_VERB_ORDER = ["SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"]


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
        runtime_tools: Optional[List[Tuple[str, str]]] = None,
        include_demo: bool = True,
        include_error_handling: bool = True,
        allowed_verbs: Optional[List[str]] = None,
        allowed_runtime_tools: Optional[List[str]] = None,
    ):
        """
        Args:
            language: 语言 ("zh" 或 "en")
            runtime_tools: 运行时工具注册表 [(alias, description), ...]
                           如果为 None，使用 DEFAULT_RUNTIME_TOOLS
            include_demo: 是否包含 One-Shot 演示
            include_error_handling: 是否包含错误恢复指令
            allowed_verbs: MTP 动词白名单，None=全量渲染
            allowed_runtime_tools: 系统工具白名单，None=全量渲染，
                                   空列表=不渲染工具列表
        """
        self.language = resolve_language(explicit=language)
        self.include_demo = include_demo
        self.include_error_handling = include_error_handling

        base_tools = runtime_tools if runtime_tools is not None else DEFAULT_RUNTIME_TOOLS
        if allowed_runtime_tools is not None:
            allowed_set = set(allowed_runtime_tools)
            self.runtime_tools = [(a, d) for a, d in base_tools if a in allowed_set]
        else:
            self.runtime_tools = base_tools

        # 权限过滤：MTP 动词白名单 (用于协议规格渲染)
        self.allowed_verbs = (
            set(v.upper() for v in allowed_verbs)
            if allowed_verbs is not None
            else None
        )

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
        if self.runtime_tools:
            sections.append(self._build_runtime_tools())

        # 5. 高密度演示 (可选)
        if self.include_demo:
            sections.append(self._build_dense_demo())

        # 6. 错误恢复 (可选)
        if self.include_error_handling:
            sections.append(self._build_error_handling())

        return "\n\n".join(sections)

    def _build_protocol_spec(self) -> str:
        """构建协议规格模块，根据 allowed_verbs 动态渲染指令集"""
        template = get_mtp_prompt_text("protocol_spec", self.language)

        verbs = [v for v in _VERB_ORDER if self.allowed_verbs is None or v in self.allowed_verbs]
        verb_lines = [f"   - {v}: {get_mtp_verb_text(v, self.language)}" for v in verbs]

        return template.format(
            left_delim=MTP_LEFT_DELIMITER,
            right_delim=MTP_RIGHT_DELIMITER,
            verb_list="\n".join(verb_lines),
        )

    def _build_negative_constraints(self) -> str:
        """构建负面约束模块"""
        return get_mtp_prompt_text("negative_constraints", self.language)

    def _build_behavioral_guidelines(self) -> str:
        """构建行为准则模块"""
        text = get_mtp_prompt_text("behavioral_guidelines", self.language)
        if self.allowed_verbs is not None and "CALL" not in self.allowed_verbs:
            return "\n".join(
                line for line in text.splitlines()
                if "CALL" not in line
            )
        return text

    def _build_runtime_tools(self) -> str:
        """构建运行时工具列表"""
        lines = []
        for alias, desc in self.runtime_tools:
            lines.append(f"- `{alias}`: {desc}")
        tool_list = "\n".join(lines)
        template = get_mtp_prompt_text("runtime_tools_template", self.language)
        return template.format(tool_list=tool_list)

    def _build_dense_demo(self) -> str:
        """构建高密度演示模块"""
        template = get_mtp_prompt_text("dense_demo", self.language)
        return template.format(
            left_delim=MTP_LEFT_DELIMITER,
            right_delim=MTP_RIGHT_DELIMITER,
        )

    def _build_error_handling(self) -> str:
        """构建错误处理指令"""
        return get_mtp_prompt_text("error_handling", self.language)


# ========== 便捷函数 ==========

def get_mtp_prompt(
    language: str = "zh",
    runtime_tools: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    便捷函数: 获取 MTP System Prompt 片段

    Args:
        language: 语言 ("zh" 或 "en")
        runtime_tools: 可用的运行时工具列表

    Returns:
        str: MTP System Prompt 片段
    """
    builder = MTPPromptBuilder(
        language=language,
        runtime_tools=runtime_tools,
    )
    return builder.build()


__all__ = [
    "MTPPromptBuilder",
    "get_mtp_prompt",
    "DEFAULT_RUNTIME_TOOLS",
]
