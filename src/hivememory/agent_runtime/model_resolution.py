"""Agent 执行侧的模型解析端口。

执行侧只需要"按模型名解析出运行所需的 LLM 配置与展示名"这一项能力；
实现方是 System 的模型注册表（``system.model_registry.ModelRegistry``），由
组合根注入。找不到模型时实现方抛出 ``core.errors.ModelNotFoundError``。
"""

from __future__ import annotations

from typing import Protocol

from hivememory.config.shared import LLMConfig


class ModelResolver(Protocol):
    """按模型名解析运行时 LLM 配置的端口。"""

    def resolve(
        self,
        model_name: str,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        top_p_override: float | None = None,
    ) -> tuple[LLMConfig, str]:
        """返回 ``(LLMConfig, display_name)``；``model_name`` 为 ``default`` 时取默认模型。"""
        ...


__all__ = ["ModelResolver"]
