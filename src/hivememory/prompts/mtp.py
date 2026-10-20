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
"""

import logging
import re
from collections.abc import Iterable

from hivememory.config.alice import KoakumaConfig
from hivememory.core.mtp.models import (
    MTP_LEFT_DELIMITER,
    MTP_RIGHT_DELIMITER,
)
from hivememory.i18n import get_mtp_prompt_text, get_mtp_verb_text, resolve_language

logger = logging.getLogger(__name__)


# ========== MVP 默认运行时工具列表 (Chapter 8.6) ==========

DEFAULT_RUNTIME_TOOLS: list[tuple[str, str]] = [
    ("sys_clock", "Get current date, time, and timezone."),
    ("sys_web_search", "Search the internet for latest information."),
    ("sys_read_file", "Read a file from the workspace."),
    ("sys_write_file", "Write content to a file in the workspace."),
    ("sys_python_repl", "Execute Python code for calculation or data processing."),
]


# 默认全量动词顺序
_VERB_ORDER = ["SEARCH", "READ", "RUN", "WRITE", "UPDATE", "CALL"]
# 教学模板中的独立动词标记；不把普通单词或系统工具别名当作 MTP 动词。
# 只以 ASCII 字母、数字与下划线判定边界：Unicode 下的 ``\b`` 把汉字也算作
# 单词字符，中文模板中紧贴汉字的动词（如“先SEARCH”）会被漏判。
_VERB_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:SEARCH|READ|RUN|WRITE|UPDATE|CALL)(?![A-Za-z0-9_])"
)
_DEMO_VERBS = frozenset({"SEARCH", "READ", "RUN"})


# ========== Prompt 构建器 ==========


class MTPPromptBuilder:
    """
    MTP 协议 System Prompt 构建器

    仅负责生成 MTP 协议教学片段。不包含角色设定（persona）、
    预检索记忆。根帧的人格与本轮材料由任务进程编译，子帧由 CALL 编排组装。

    使用示例:
        >>> builder = MTPPromptBuilder(language="en")
        >>> mtp_fragment = builder.build()
    """

    def __init__(
        self,
        language: str = "zh",
        runtime_tools: list[tuple[str, str]] | None = None,
        include_demo: bool = True,
        include_error_handling: bool = True,
        allowed_verbs: Iterable[str] | None = None,
    ):
        """
        Args:
            language: 语言 ("zh" 或 "en")
            runtime_tools: 运行时工具注册表 [(alias, description), ...]
                           如果为 None，使用 DEFAULT_RUNTIME_TOOLS
            include_demo: 是否包含 One-Shot 演示
            include_error_handling: 是否包含错误恢复指令
            allowed_verbs: frame 可用的 MTP 动词集合，None=全量渲染；
                           只约束教学内容，不承担资源操作授权
        """
        self.language = resolve_language(explicit=language)
        self.include_demo = include_demo
        self.include_error_handling = include_error_handling

        # 系统工具不设 agent 权限，菜单完整展示当前提供的工具列表。
        self.runtime_tools = runtime_tools if runtime_tools is not None else DEFAULT_RUNTIME_TOOLS

        # 同一集合同时裁剪协议规格、行为教学和演示，避免其它段落鼓励禁用动词。
        self.allowed_verbs = (
            frozenset(v.upper() for v in allowed_verbs) if allowed_verbs is not None else None
        )

    def build(self) -> str:
        """
        构建 MTP 协议教学 System Prompt 片段

        Returns:
            str: MTP 协议教学文本
        """
        sections = []

        # 1. 协议规格 (含 HIVE MEMORY 上下文 header)
        sections.append(self._build_protocol_spec())

        # 2. 负面约束
        sections.append(self._build_negative_constraints())

        # 3. 行为准则
        sections.append(self._build_behavioral_guidelines())

        # 4. 系统工具列表 (有工具时渲染)
        if self.runtime_tools:
            sections.append(self._build_runtime_tools())

        # 5. 高密度演示 (可选)
        if self.include_demo and (self.allowed_verbs is None or _DEMO_VERBS <= self.allowed_verbs):
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
        return self._filter_verb_guidance(
            get_mtp_prompt_text("negative_constraints", self.language)
        )

    def _build_behavioral_guidelines(self) -> str:
        """构建行为准则模块"""
        return self._filter_verb_guidance(
            get_mtp_prompt_text("behavioral_guidelines", self.language)
        )

    def _filter_verb_guidance(self, text: str) -> str:
        """仅保留所提及动词全部可用的教学行，避免裁剪后留下误导性的建议。"""
        if self.allowed_verbs is None:
            return text
        return "\n".join(
            line
            for line in text.splitlines()
            if set(_VERB_PATTERN.findall(line)) <= self.allowed_verbs
        )

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
        return self._filter_verb_guidance(get_mtp_prompt_text("error_handling", self.language))


# ========== 便捷函数 ==========


def get_mtp_prompt(
    language: str = "zh",
    runtime_tools: list[tuple[str, str]] | None = None,
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


def render_mtp_teaching(
    config: KoakumaConfig,
    *,
    language: str,
    permitted_verbs: Iterable[str],
) -> str:
    """按 Koakuma 配置与帧的动词集合生成教学；Koakuma 或教学开关关闭时不教学。

    根帧与 CALL 子帧共用这一规则，各自只传入自己的语言与动词集合。
    """
    prompt_config = config.mtp_prompt
    if not (config.enabled and prompt_config.enabled):
        return ""
    return MTPPromptBuilder(
        language=language,
        include_demo=prompt_config.include_demo,
        include_error_handling=prompt_config.include_error_handling,
        allowed_verbs=permitted_verbs,
    ).build()


__all__ = [
    "MTPPromptBuilder",
    "get_mtp_prompt",
    "render_mtp_teaching",
    "DEFAULT_RUNTIME_TOOLS",
]
