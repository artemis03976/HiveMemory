"""
MTP (Memory Tool Protocol) 解析器与格式化器

定义 MTP 协议的解析、验证和响应格式化逻辑。

协议语法: ⟪ VERB | TARGET | ARGS ⟫
- VERB: SEARCH, READ, RUN, WRITE, UPDATE
- TARGET: *, alias, [alias1, alias2]
- ARGS: key="value" 或 key=`raw content`

对应设计文档: MemoryToolProtocol.md Chapter 2 & 3.3

作者: HiveMemory Team
版本: 1.0
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from hivememory.core.models import MemoryType
from hivememory.engines.retrieval.models import QueryFilters

logger = logging.getLogger(__name__)

# ========== 协议常量 ==========

# MTP 定界符 (Section 2.1)
MTP_LEFT_DELIMITER = "\u27EA"   # ⟪ (U+27EA, MATHEMATICAL LEFT DOUBLE ANGLE BRACKET)
MTP_RIGHT_DELIMITER = "\u27EB"  # ⟫ (U+27EB, MATHEMATICAL RIGHT DOUBLE ANGLE BRACKET)
MTP_SEPARATOR = "|"

# Stop Sequence: 在调用 LLM API 时设置 stop=["⟫"] (Section 3.1.1)
MTP_STOP_SEQUENCE = MTP_RIGHT_DELIMITER


# ========== 枚举定义 ==========

class MTPVerb(str, Enum):
    """
    MTP 指令动词枚举 (Section 2.2)

    核心能力 (同步):
        SEARCH - 发现，模糊检索返回 Index 菜单
        READ   - 查阅，获取记忆原子的 Payload 内容
        RUN    - 执行，调用内核工具或记忆中的代码

    特权信号 (异步):
        WRITE  - 记录，向帕秋莉发送高优先级保存信号
        UPDATE - 修正，请求更新已有记忆
    """
    SEARCH = "SEARCH"
    READ = "READ"
    RUN = "RUN"
    WRITE = "WRITE"
    UPDATE = "UPDATE"


class MTPResponseStatus(str, Enum):
    """
    MTP 响应状态枚举 (Section 3.3.2)

    SUCCESS - 执行成功
    ERROR   - 执行失败
    ACK     - 异步信号已确认 (用于 WRITE/UPDATE)
    WARNING - 部分成功，附带警告 (如 filter 降级)
    """
    SUCCESS = "success"
    ERROR = "error"
    ACK = "ack"
    WARNING = "warning"


# ========== 数据模型 ==========

class MTPTarget(BaseModel):
    """
    MTP 指令目标模型 (Section 2.1)

    支持三种形态:
    - 全局通配: "*" 或 "global"
    - 单别名: "fact_api_spec"
    - 列表: ["fact_api_spec", "tool_db_connector"]
    """
    is_wildcard: bool = Field(default=False, description="是否为全局通配")
    aliases: List[str] = Field(default_factory=list, description="别名列表")

    @property
    def is_list(self) -> bool:
        """是否为列表目标"""
        return len(self.aliases) > 1

    @property
    def single_alias(self) -> Optional[str]:
        """获取单个别名（非列表且非通配时）"""
        if not self.is_wildcard and len(self.aliases) == 1:
            return self.aliases[0]
        return None


class MTPCommand(BaseModel):
    """
    MTP 指令数据模型 (Section 2.1)

    解析后的完整 MTP 指令，包含动词、目标和参数。

    Attributes:
        verb: 指令动词 (SEARCH/READ/RUN/WRITE/UPDATE)
        target: 指令目标
        args: 参数字典 (key-value pairs)
        raw_text: 原始指令文本
    """
    verb: MTPVerb = Field(..., description="指令动词")
    target: MTPTarget = Field(default_factory=MTPTarget, description="指令目标")
    args: Dict[str, str] = Field(default_factory=dict, description="参数字典")
    raw_text: str = Field(default="", description="原始指令文本")


class MTPResponse(BaseModel):
    """
    MTP 响应数据模型 (Section 3.3.2)

    内核执行结果的结构化表示，用于生成 XML 响应容器。

    Attributes:
        status: 响应状态
        content: 响应内容 (markdown 或自然语言)
        execution_time_ms: 执行耗时 (毫秒)
    """
    status: MTPResponseStatus = Field(..., description="响应状态")
    content: str = Field(default="", description="响应内容")
    execution_time_ms: float = Field(default=0.0, description="执行耗时 (毫秒)")


# ========== 异常定义 ==========

# MTPParseError 定义在 exceptions.py 中，此处导入以保持向后兼容
from hivememory.patchouli.protocol.exceptions import MTPParseError  # noqa: F401


# ========== MTP 解析器 ==========

class MTPParser:
    """
    MTP 协议解析器 (Section 2.1)

    负责将原始 MTP 指令文本解析为结构化的 MTPCommand 对象。

    解析规则:
    1. 提取 ⟪...⟫ 之间的内容
    2. 按前两个 | 分割为 VERB, TARGET, ARGS 三段
    3. 解析 TARGET: *, alias, [alias1, alias2]
    4. 解析 ARGS: key="value" 和 key=`raw content`

    使用示例:
        >>> parser = MTPParser()
        >>> cmd = parser.parse('⟪ READ | fact_api_spec | ⟫')
        >>> cmd.verb == MTPVerb.READ
        True
    """

    # 提取定界符之间的内容
    _COMMAND_PATTERN = re.compile(
        r"\u27EA\s*(.*?)\s*\u27EB",
        re.DOTALL,
    )

    # 解析 key="value" 参数
    _KV_PATTERN = re.compile(
        r'(\w+)\s*=\s*"([^"]*)"'
    )

    # 解析 key=`raw content` 参数 (反引号包裹，支持多行)
    _RAW_PATTERN = re.compile(
        r'(\w+)\s*=\s*`(.*?)`',
        re.DOTALL,
    )

    # 解析列表目标 [a, b, c]
    _LIST_TARGET_PATTERN = re.compile(
        r'\[\s*([\w\s,]+)\s*\]'
    )

    # 合法动词集合
    _VALID_VERBS = {v.value for v in MTPVerb}

    def parse(self, text: str) -> MTPCommand:
        """
        解析 MTP 指令文本

        Args:
            text: 原始指令文本 (包含定界符 ⟪...⟫)

        Returns:
            MTPCommand: 解析后的指令对象

        Raises:
            MTPParseError: 解析失败
        """
        match = self._COMMAND_PATTERN.search(text)
        if not match:
            raise MTPParseError(
                f"No MTP command found. Expected '{MTP_LEFT_DELIMITER}...{MTP_RIGHT_DELIMITER}'"
            )

        inner = match.group(1).strip()
        raw_text = match.group(0)

        verb_str, target_str, args_str = self._split_segments(inner)

        # 验证动词
        verb_upper = verb_str.upper()
        if verb_upper not in self._VALID_VERBS:
            raise MTPParseError(
                f"Unknown verb '{verb_str}'. "
                f"Valid verbs: {', '.join(sorted(self._VALID_VERBS))}"
            )

        verb = MTPVerb(verb_upper)
        target = self._parse_target(target_str)
        args = self._parse_args(args_str)

        return MTPCommand(
            verb=verb,
            target=target,
            args=args,
            raw_text=raw_text,
        )
# PLACEHOLDER_PARSER_METHODS

    def complete_and_parse(self, text: str) -> MTPCommand:
        """
        补全并解析不完整的 MTP 指令 (Section 3.1.2)

        用于 Stop Sequence 场景: LLM 在 ⟫ 处被截断，
        文本可能不含闭合定界符。自动追加 ⟫ 后解析。

        Args:
            text: 被截断的文本 (可能不含 ⟫)

        Returns:
            MTPCommand: 解析后的指令对象

        Raises:
            MTPParseError: 解析失败
        """
        if MTP_RIGHT_DELIMITER not in text:
            text = text.rstrip() + " " + MTP_RIGHT_DELIMITER
        return self.parse(text)

    def detect_command(self, text: str) -> bool:
        """
        检测文本中是否包含 MTP 指令

        Args:
            text: 待检测文本

        Returns:
            bool: 是否包含 MTP 指令 (至少包含左定界符 ⟪)
        """
        return MTP_LEFT_DELIMITER in text

    def _split_segments(self, inner: str) -> Tuple[str, str, str]:
        """
        按前两个 | 分割为三段 (Section 2.1)

        关键规则: 仅前两个 | 是分隔符，ARGS 内部的 | 视为内容。

        Args:
            inner: 定界符内部的文本

        Returns:
            (verb_str, target_str, args_str) 三元组

        Raises:
            MTPParseError: 缺少分隔符
        """
        first_pipe = inner.find(MTP_SEPARATOR)
        if first_pipe == -1:
            raise MTPParseError(
                f"Missing separator '{MTP_SEPARATOR}' in MTP command"
            )

        verb_str = inner[:first_pipe].strip()
        rest = inner[first_pipe + 1:]

        second_pipe = rest.find(MTP_SEPARATOR)
        if second_pipe == -1:
            # 仅两段: VERB | TARGET (无 ARGS)
            target_str = rest.strip()
            args_str = ""
        else:
            target_str = rest[:second_pipe].strip()
            args_str = rest[second_pipe + 1:].strip()

        return verb_str, target_str, args_str

    def _parse_target(self, target_str: str) -> MTPTarget:
        """
        解析 TARGET 字段 (Section 2.1)

        支持:
        - "*" 或 "global" -> 通配
        - "[a, b, c]" -> 列表
        - "alias_name" -> 单别名

        Args:
            target_str: TARGET 字段文本

        Returns:
            MTPTarget 对象
        """
        target_str = target_str.strip()

        if target_str in ("*", "global"):
            return MTPTarget(is_wildcard=True)

        # 列表目标: [alias1, alias2, alias3]
        list_match = self._LIST_TARGET_PATTERN.match(target_str)
        if list_match:
            items = list_match.group(1)
            aliases = [a.strip() for a in items.split(",") if a.strip()]
            return MTPTarget(aliases=aliases)

        # 单别名
        if target_str:
            return MTPTarget(aliases=[target_str])

        return MTPTarget()
# PLACEHOLDER_PARSE_ARGS

    def _parse_args(self, args_str: str) -> Dict[str, str]:
        """
        解析 ARGS 字段 (Section 2.1)

        支持两种格式:
        - key="value" (双引号包裹)
        - key=`raw content` (反引号包裹，支持多行)

        Args:
            args_str: ARGS 字段文本

        Returns:
            参数字典
        """
        if not args_str:
            return {}

        args: Dict[str, str] = {}

        # 先解析反引号参数 (可能包含双引号)
        for match in self._RAW_PATTERN.finditer(args_str):
            key, value = match.group(1), match.group(2)
            args[key] = value.strip()

        # 移除已解析的反引号参数，再解析双引号参数
        remaining = self._RAW_PATTERN.sub("", args_str)
        for match in self._KV_PATTERN.finditer(remaining):
            key, value = match.group(1), match.group(2)
            if key not in args:
                args[key] = value

        return args


# ========== MTP Filter 解析器 ==========

# MTP filter "type:XXX" 值到 MemoryType 枚举的映射 (大小写不敏感)
_FILTER_TYPE_MAP: Dict[str, MemoryType] = {
    "code": MemoryType.CODE_SNIPPET,
    "code_snippet": MemoryType.CODE_SNIPPET,
    "fact": MemoryType.FACT,
    "url": MemoryType.URL_RESOURCE,
    "url_resource": MemoryType.URL_RESOURCE,
    "reflection": MemoryType.REFLECTION,
    "profile": MemoryType.USER_PROFILE,
    "user_profile": MemoryType.USER_PROFILE,
    "wip": MemoryType.WORK_IN_PROGRESS,
    "work_in_progress": MemoryType.WORK_IN_PROGRESS,
}

class MTPFilterParser:
    """
    MTP Filter 解析器

    专门用于解析 SEARCH 指令中的 filter 参数字符串。
    """

    def parse(self, filter_str: str) -> Tuple[Optional[QueryFilters], List[str]]:
        """
        解析 MTP SEARCH 指令的 filter 参数 (Section 2.2)

        语法: key:value 对，多个用空格分隔
        支持的 key:
            - type: 记忆类型 (CODE, FACT, URL, REFLECTION, PROFILE, WIP)
            - tag: 标签 (可多次出现)
            - agent: 来源 Agent ID
            - confidence: 最小置信度 (0.0-1.0)

        安全策略: 宽容解析，降级并返回警告
            - 无法识别的 key/value → 忽略 + 返回警告
            - 解析后全空 → 返回 (None, warnings)
            - 任何异常 → 返回 (None, warnings)

        Args:
            filter_str: 原始 filter 字符串，如 "type:CODE" 或 "type:FACT tag:python"

        Returns:
            Tuple[Optional[QueryFilters], List[str]]: (过滤条件, 警告列表)
        """
        if not filter_str or not filter_str.strip():
            return None, []

        warnings: List[str] = []

        try:
            memory_type = None
            tags: List[str] = []
            source_agent_id = None
            min_confidence = 0.0

            for token in filter_str.strip().split():
                if ":" not in token:
                    warnings.append(f"Note: Filter token '{token}' was ignored (missing ':' separator).")
                    logger.warning(f"MTP filter: 忽略无法解析的 token '{token}'")
                    continue

                key, _, value = token.partition(":")
                key = key.strip().lower()
                value = value.strip()

                if not key or not value:
                    warnings.append(f"Note: Filter token '{token}' was ignored (empty key or value).")
                    logger.warning(f"MTP filter: 忽略空 key 或 value: '{token}'")
                    continue

                if key == "type":
                    mapped = _FILTER_TYPE_MAP.get(value.lower())
                    if mapped is not None:
                        memory_type = mapped
                    else:
                        warnings.append(
                            f"Note: Unknown filter type '{value}' was ignored. "
                            f"Valid types: CODE, FACT, URL, REFLECTION, PROFILE, WIP."
                        )
                        logger.warning(
                            f"MTP filter: 未知 type 值 '{value}'，已忽略。"
                            f"支持: CODE, FACT, URL, REFLECTION, PROFILE, WIP"
                        )
                elif key == "tag":
                    tags.append(value)
                elif key == "agent":
                    source_agent_id = value
                elif key == "confidence":
                    try:
                        parsed = float(value)
                        if 0.0 < parsed <= 1.0:
                            min_confidence = parsed
                        else:
                            warnings.append(
                                f"Note: Filter confidence value {parsed} is out of range (0,1] and was ignored."
                            )
                            logger.warning(
                                f"MTP filter: confidence 值 {parsed} 超出范围 (0,1]，已忽略"
                            )
                    except ValueError:
                        warnings.append(
                            f"Note: Filter confidence value '{value}' is not a valid number and was ignored."
                        )
                        logger.warning(
                            f"MTP filter: confidence 值 '{value}' 不是有效数字，已忽略"
                        )
                else:
                    warnings.append(f"Note: Unknown filter key '{key}' was ignored.")
                    logger.warning(f"MTP filter: 未知 key '{key}'，已忽略")

            # 构建 QueryFilters，全空则返回 None
            filters = QueryFilters(
                memory_type=memory_type,
                tags=tags,
                source_agent_id=source_agent_id,
                min_confidence=min_confidence,
            )

            if filters.is_empty():
                return None, warnings

            logger.info(f"MTP filter 解析结果: {filters}")
            return filters, warnings

        except Exception as e:
            logger.warning(f"MTP filter 解析异常，已降级为无 filter: {e}")
            warnings.append("Note: Filter parsing failed. Results may be broader than expected.")
            return None, warnings


# ========== MTP 格式化器 ==========

class MTPFormatter:
    """
    MTP 响应格式化器 (Section 3.3)

    将执行结果格式化为 XML 响应容器，用于注入 Assistant 历史。

    响应容器格式 (Section 3.3.2):
        <mtp_response status="success|error" time="ms">
            ... Payload ...
        </mtp_response>
    """

    @staticmethod
    def format_response(response: MTPResponse) -> str:
        """
        格式化 MTP 响应为 XML 容器

        Args:
            response: MTP 响应对象

        Returns:
            str: XML 格式的响应字符串
        """
        time_attr = ""
        if response.execution_time_ms > 0:
            time_attr = f' time="{response.execution_time_ms:.0f}ms"'

        return (
            f'<mtp_response status="{response.status.value}"{time_attr}>\n'
            f'{response.content}\n'
            f'</mtp_response>'
        )

    @staticmethod
    def format_command_with_response(
        command: MTPCommand,
        response: MTPResponse,
    ) -> str:
        """
        格式化完整的 指令 + 响应 文本块 (Section 3.3.1)

        用于 Fake Assistant History 回填:
        ⟪ VERB | TARGET | ARGS ⟫
        <mtp_response>...</mtp_response>

        Args:
            command: 原始指令
            response: 执行响应

        Returns:
            str: 完整的回填文本
        """
        time_attr = ""
        if response.execution_time_ms > 0:
            time_attr = f' time="{response.execution_time_ms:.0f}ms"'

        response_xml = (
            f'<mtp_response status="{response.status.value}"{time_attr}>\n'
            f'{response.content}\n'
            f'</mtp_response>'
        )

        return f"{command.raw_text}\n{response_xml}"


# ========== 工厂函数 ==========

def create_parser() -> MTPParser:
    """创建 MTP 解析器实例"""
    return MTPParser()


def create_formatter() -> MTPFormatter:
    """创建 MTP 格式化器实例"""
    return MTPFormatter()


def create_filter_parser() -> MTPFilterParser:
    """创建 MTP Filter 解析器实例"""
    return MTPFilterParser()


__all__ = [
    # 常量
    "MTP_LEFT_DELIMITER",
    "MTP_RIGHT_DELIMITER",
    "MTP_STOP_SEQUENCE",
    "MTP_SEPARATOR",
    # 枚举
    "MTPVerb",
    "MTPResponseStatus",
    # 数据模型
    "MTPTarget",
    "MTPCommand",
    "MTPResponse",
    # 解析器
    "MTPParser",
    "MTPFilterParser",
    "MTPParseError",
    # 格式化器
    "MTPFormatter",
    # 工厂函数
    "create_parser",
    "create_filter_parser",
    "create_formatter",
]
