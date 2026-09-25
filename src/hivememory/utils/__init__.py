"""
HiveMemory 工具模块集合。

项目各处共用的工具类与工具函数。
"""

from hivememory.i18n import Language
from hivememory.utils.json_parser import (
    JSONParseError,
    LLMJSONParser,
    parse_llm_json,
    parse_llm_json_many,
    safe_parse_llm_json,
)
from hivememory.utils.time import require_utc, utc_now
from hivememory.utils.time_formatter import (
    TimeFormatter,
    format_time_ago,
)
from hivememory.utils.token_estimator import (
    EstimationStrategy,
    TokenEstimator,
    estimate_tokens,
)
from hivememory.utils.uuid import normalize_uuid

__all__ = [
    "TimeFormatter",
    "Language",
    "format_time_ago",
    "utc_now",
    "require_utc",
    "LLMJSONParser",
    "JSONParseError",
    "parse_llm_json",
    "parse_llm_json_many",
    "safe_parse_llm_json",
    "TokenEstimator",
    "EstimationStrategy",
    "estimate_tokens",
    "normalize_uuid",
]
