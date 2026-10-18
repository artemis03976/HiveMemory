"""Agent 执行侧的模型解析端口。

执行侧只需要"按模型名解析出运行所需的 LLM 配置与展示名"这一项能力；
实现方是 System 的模型注册表（``system.model_registry.ModelRegistry``），由
组合根注入。找不到模型时实现方抛出 ``core.errors.ModelNotFoundError``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from hivememory.config.alice import AgentRuntimeConfig
from hivememory.config.shared import LLMConfig
from hivememory.core.constants import DEFAULT_MAX_TOKENS
from hivememory.core.models import AgentProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedFrameModel:
    """一次解析得到的生成参数与独立输入预算，供同一帧恢复时复用。"""

    generation_options: dict[str, Any]
    input_token_limit: int
    display_name: str = ""


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


def resolve_frame_model(
    profile: AgentProfile,
    generation_options: dict[str, Any] | None,
    *,
    model_registry: ModelResolver | None,
    config: AgentRuntimeConfig,
) -> ResolvedFrameModel:
    """
    根据 agent_profile 从注册表解析 LLM 参数，注入到 generation_options
    供 LLMClient 使用；输入预算与展示名独立返回，不传给模型 SDK。

    参数覆盖优先级（从高到低）：
        会话请求 (generation_options) > Agent Profile > 模型注册表定义

    - model：会话传入的注册表 ID 覆盖 profile.model_name（供侧边栏实时切换模型调试）
    - api_key / api_base：来自解析出的模型（Profile 仅通过 model_name 选择模型）
    - temperature / top_p：会话覆盖优先，其次 Profile，最后模型定义默认
    - max_tokens：会话覆盖优先，其次模型定义默认

    设计原则：
    - 注册表未注入（model_registry 为 None）时，由调用方通过 generation_options
      直接传 model；LLMClient 缺 model 时会抛出明确错误。
    - 注册表解析失败由 run_frame 稳定投影为 FAILED，不静默降级到其他模型。
    """
    session_opts = dict(generation_options or {})
    if model_registry is None:
        max_tokens = session_opts.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        return ResolvedFrameModel(
            generation_options=session_opts,
            input_token_limit=_input_token_limit(config, None, max_tokens),
        )

    # 模型选择：会话传入的注册表 ID 优先，否则用 Agent Profile 的 model_name。
    # 二者语义一致（都是注册表 ID / 'default'），resolve 统一解析。
    session_model_id = session_opts.get("model")
    effective_model_name = session_model_id or profile.model_name

    # 覆盖值：会话请求优先于 Agent Profile；两者皆无则由 resolve 回落到模型定义
    temperature_override = session_opts.get("temperature")
    if temperature_override is None:
        temperature_override = profile.temperature

    top_p_override = session_opts.get("top_p")
    if top_p_override is None:
        top_p_override = profile.top_p

    max_tokens_override = session_opts.get("max_tokens")

    # 解析失败由 run_frame 转为 frame FAILED，由编排执行器按 root/callee 关系解释。
    llm_config, display_name = model_registry.resolve(
        effective_model_name,
        temperature_override=temperature_override,
        max_tokens_override=max_tokens_override,
        top_p_override=top_p_override,
    )

    # 用解析结果覆盖 generation_options 中的对应键——
    # resolve 已完成三级优先级合并，这里以其结果为准（包括把会话传入的
    # 注册表 ID 替换为 litellm 模型标识符，供 LLMClient 使用）。
    # api_key / api_base 为 None 是合法状态（litellm 从环境变量读取）。
    resolved: dict[str, Any] = {
        **session_opts,
        "model": llm_config.model,
        "api_key": llm_config.api_key,
        "api_base": llm_config.api_base,
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
        "top_p": llm_config.top_p,
    }
    logger.debug(
        f"模型解析: effective={effective_model_name!r} "
        f"(session={session_model_id!r}, profile={profile.model_name!r}) → "
        f"litellm_model={llm_config.model!r}, display={display_name!r}, "
        f"temperature={llm_config.temperature}, top_p={llm_config.top_p}"
    )
    return ResolvedFrameModel(
        generation_options=resolved,
        input_token_limit=_input_token_limit(
            config, llm_config.context_window, llm_config.max_tokens
        ),
        display_name=display_name,
    )


def _input_token_limit(
    config: AgentRuntimeConfig, context_window: int | None, max_tokens: int
) -> int:
    """生成预算和安全余量从总窗口扣除；非正预算保留给输入检查明确拒绝。"""
    window = context_window or config.default_context_window
    return window - max_tokens - config.safety_margin
