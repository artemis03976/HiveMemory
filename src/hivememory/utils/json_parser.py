"""
HiveMemory - JSON 解析工具 (JSON Parser Utility)

职责:
    提供高容错的 JSON 解析功能，专门用于解析 LLM 返回的各种格式输出。

特性:
    - 多策略解析: Markdown 代码块、纯 JSON、智能提取
    - 容错处理: 支持 Python 字典风格（单引号、True/False/None）
    - 括号匹配: 智能提取完整的 JSON 对象
    - 类型推断: 自动返回 dict 或 list

作者: HiveMemory Team
版本: 0.1.0
"""

import json
import ast
import logging
import re
from typing import Any, Optional, TypeVar, Type, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class JSONParseError(Exception):
    """JSON 解析失败异常"""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class LLMJSONParser:
    """
    LLM JSON 输出解析器

    专门用于解析 LLM 返回的 JSON，提供高容错解析能力。

    解析策略（按优先级）:
        1. Markdown 代码块提取 (支持 ```json, ```JSON, ``` (无语言))
        2. 直接解析全文（如果全文本身就是 JSON）
        3. 智能提取第一个 JSON 对象（基于括号计数）
        4. 容错处理：支持单引号、尾部逗号、True/False/None（通过 ast.literal_eval）

    Examples:
        >>> parser = LLMJSONParser()
        >>>
        >>> # 解析 Markdown 代码块中的 JSON
        >>> result = parser.parse('''
        ...     ```json
        ...     {"name": "Alice", "age": 30}
        ...     ```
        ... ''')
        >>> print(result)  # {'name': 'Alice', 'age': 30}
        >>>
        >>> # 解析纯 JSON
        >>> result = parser.parse('{"key": "value"}')
        >>>
        >>> # 从混合文本中提取 JSON
        >>> result = parser.parse('''
        ...     这是分析结果：{"summary": "成功"}，完成。
        ... ''')
        >>>
        >>> # 解析为 Pydantic 模型
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> user = parser.parse('{"name": "Bob", "age": 25}', as_model=User)
        >>> assert user.name == "Bob"
        >>>
        >>> # 批量解析多个 JSON 对象
        >>> results = parser.parse_many('''
        ...     {"id": 1}
        ...     {"id": 2}
        ... ''')
    """

    # 正则表达式模式
    MARKDOWN_CODE_BLOCK_PATTERN = r'```(?:json|JSON)?\s*(\{.*?\}|\[.*?\])\s*```'

    def __init__(
        self,
        strict: bool = False,
        strip_bom: bool = True,
        log_errors: bool = True,
    ):
        """
        初始化解析器

        Args:
            strict: 严格模式（开启时只接受标准 JSON）
            strip_bom: 是否移除 BOM 头
            log_errors: 是否记录解析错误日志
        """
        self.strict = strict
        self.strip_bom = strip_bom
        self.log_errors = log_errors

    def parse(
        self,
        raw_output: str,
        as_model: Optional[Type[T]] = None,
        default: Any = None,
    ) -> Optional[T]:
        """
        解析 JSON 字符串

        Args:
            raw_output: 待解析的原始字符串
            as_model: 可选，将解析结果转换为指定的 Pydantic 模型
            default: 解析失败时返回的默认值

        Returns:
            解析后的结果（dict/list 或指定的模型类型），失败时返回 default

        Raises:
            JSONParseError: 当 default=None 且解析失败时抛出
        """
        try:
            result = self._parse_with_strategies(raw_output)

            if result is None:
                if default is not None:
                    return default
                raise JSONParseError("无法解析 JSON 输出", raw_output)

            if as_model is not None:
                return as_model(**result)

            return result

        except JSONParseError:
            if default is not None:
                return default
            raise
        except Exception as e:
            if self.log_errors:
                logger.debug(f"模型转换失败: {e}")
            if default is not None:
                return default
            raise JSONParseError(f"模型转换失败: {e}", raw_output) from e

    def parse_many(
        self,
        raw_output: str,
        as_model: Optional[Type[T]] = None,
    ) -> list[T]:
        """
        批量解析多个 JSON 对象

        Args:
            raw_output: 待解析的原始字符串（可能包含多个 JSON 对象）
            as_model: 可选，将解析结果转换为指定的 Pydantic 模型

        Returns:
            解析后的对象列表
        """
        results = []

        for candidate in self._extract_all_json_candidates(raw_output):
            parsed = self._try_parse_json_string(candidate)
            if parsed is not None:
                if as_model is not None:
                    try:
                        results.append(as_model(**parsed))
                    except Exception as e:
                        if self.log_errors:
                            logger.debug(f"模型转换失败: {e}")
                else:
                    results.append(parsed)

        return results

    def safe_parse(self, raw_output: str) -> Optional[dict]:
        """
        安全解析（始终返回 dict 或 None，不抛出异常）

        Args:
            raw_output: 待解析的原始字符串

        Returns:
            解析后的字典，失败时返回 None
        """
        return self.parse(raw_output, default=None)

    def _parse_with_strategies(self, raw_output: str) -> Optional[dict]:
        """
        使用多种策略解析 JSON

        Args:
            raw_output: 待解析的原始字符串

        Returns:
            解析后的字典，所有策略失败时返回 None
        """
        # 预处理
        text = self._preprocess(raw_output)

        # 按策略顺序尝试解析
        for candidate in self._generate_candidates(text):
            result = self._try_parse_json_string(candidate)
            if result is not None:
                return result

        return None

    def _preprocess(self, text: str) -> str:
        """
        预处理文本

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        text = text.strip()

        # 移除 BOM 头
        if self.strip_bom and text.startswith('\ufeff'):
            text = text[1:]

        return text

    def _generate_candidates(self, text: str):
        """
        生成待尝试的候选 JSON 字符串

        Args:
            text: 预处理后的文本

        Yields:
            候选 JSON 字符串
        """
        # 策略 A: Markdown 代码块提取
        for match in re.finditer(self.MARKDOWN_CODE_BLOCK_PATTERN, text, re.DOTALL):
            candidate = match.group(1).strip()
            if candidate:
                yield candidate

        # 策略 B: 原文（如果原文本身就是 JSON）
        yield text

        # 策略 C: 智能提取第一个 JSON 对象/数组
        for candidate in self._extract_by_bracket_matching(text):
            if candidate != text:
                yield candidate

    def _extract_by_bracket_matching(self, text: str) -> list[str]:
        """
        基于括号匹配提取 JSON 对象或数组

        Args:
            text: 待提取的文本

        Returns:
            提取到的 JSON 字符串列表
        """
        results = []
        # 提取对象 {...}
        obj_result = self._extract_bracket_block(text, '{', '}')
        if obj_result:
            results.append(obj_result)
        # 提取数组 [...]
        arr_result = self._extract_bracket_block(text, '[', ']')
        if arr_result:
            results.append(arr_result)

        return results

    def _extract_bracket_block(
        self,
        text: str,
        open_char: str,
        close_char: str,
    ) -> Optional[str]:
        """
        提取括号包围的完整块

        Args:
            text: 待提取的文本
            open_char: 开括号字符
            close_char: 闭括号字符

        Returns:
            提取到的完整块，未找到时返回 None
        """
        start_idx = text.find(open_char)
        if start_idx == -1:
            return None

        bracket_count = 0
        in_string = False
        escape = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == '"' and not escape:
                in_string = not in_string

            if not in_string:
                if char == open_char:
                    bracket_count += 1
                elif char == close_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[start_idx:i + 1]

            if char == '\\':
                escape = not escape
            else:
                escape = False

        return None

    def _extract_all_json_candidates(self, text: str) -> list[str]:
        """
        提取文本中所有可能的 JSON 字符串

        Args:
            text: 待提取的文本

        Returns:
            所有候选 JSON 字符串
        """
        text = self._preprocess(text)
        candidates = []

        # 提取 Markdown 代码块中的 JSON
        for match in re.finditer(self.MARKDOWN_CODE_BLOCK_PATTERN, text, re.DOTALL):
            candidate = match.group(1).strip()
            if candidate:
                candidates.append(candidate)

        # 提取所有 JSON 对象
        obj_result = self._extract_bracket_block(text, '{', '}')
        if obj_result:
            candidates.append(obj_result)

        # 提取所有 JSON 数组
        arr_result = self._extract_bracket_block(text, '[', ']')
        if arr_result and arr_result != obj_result:
            candidates.append(arr_result)

        # 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        return unique_candidates

    def _try_parse_json_string(self, json_str: str) -> Optional[Any]:
        """
        尝试解析 JSON 字符串

        Args:
            json_str: 待解析的 JSON 字符串

        Returns:
            解析后的结果（dict/list），失败时返回 None
        """
        if not json_str or not json_str.strip():
            return None

        # 策略 1: 标准 JSON 解析
        try:
            result = json.loads(json_str)
            if isinstance(result, (dict, list)):
                return result
        except json.JSONDecodeError:
            pass

        # 严格模式下不尝试容错解析
        if self.strict:
            return None

        # 策略 2: 使用 ast.literal_eval 处理 Python 风格
        # 支持: 单引号、True/False/None、尾部逗号
        try:
            result = ast.literal_eval(json_str)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass

        return None


# 全局单例解析器
_default_parser = LLMJSONParser()


def parse_llm_json(
    raw_output: str,
    as_model: Optional[Type[T]] = None,
    default: Any = None,
) -> Optional[T]:
    """
    便捷函数：解析 LLM 返回的 JSON

    Args:
        raw_output: 待解析的原始字符串
        as_model: 可选，将解析结果转换为指定的 Pydantic 模型
        default: 解析失败时返回的默认值

    Returns:
        解析后的结果，失败时返回 default

    Examples:
        >>> result = parse_llm_json('{"name": "Alice"}')
        >>> print(result)  # {'name': 'Alice'}
        >>>
        >>> # 解析失败时返回默认值
        >>> result = parse_llm_json('not json', default={})
        >>> print(result)  # {}
    """
    return _default_parser.parse(raw_output, as_model=as_model, default=default)


def parse_llm_json_many(
    raw_output: str,
    as_model: Optional[Type[T]] = None,
) -> list[T]:
    """
    便捷函数：批量解析多个 JSON 对象

    Args:
        raw_output: 待解析的原始字符串
        as_model: 可选，将解析结果转换为指定的 Pydantic 模型

    Returns:
        解析后的对象列表
    """
    return _default_parser.parse_many(raw_output, as_model=as_model)


def safe_parse_llm_json(raw_output: str) -> Optional[dict]:
    """
    便捷函数：安全解析（始终返回 dict 或 None）

    Args:
        raw_output: 待解析的原始字符串

    Returns:
        解析后的字典，失败时返回 None
    """
    return _default_parser.safe_parse(raw_output)


__all__ = [
    "LLMJSONParser",
    "JSONParseError",
    "parse_llm_json",
    "parse_llm_json_many",
    "safe_parse_llm_json",
]
