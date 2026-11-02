"""Gateway 分析 Engine 共用的 LLM 请求参数派生。"""

from __future__ import annotations

from hivememory.infrastructure.llm.base import LLMRequestParams


def json_analysis_params(params: LLMRequestParams | None) -> LLMRequestParams | None:
    """由 Gateway 组件 LLM 参数派生分析请求：固定要求 JSON 输出。

    模型、凭证与采样参数沿用组件参数；未配置组件参数时返回 ``None``，由 Engine
    报告能力缺失。
    """
    if params is None:
        return None
    return params.model_copy(update={"json_output": True})


__all__ = ["json_analysis_params"]
