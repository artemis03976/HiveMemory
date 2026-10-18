"""Agent 执行侧的模型解析端口，以及帧模型到 LLM 请求参数的转换。

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
from hivememory.core.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import AgentProfile
from hivememory.infrastructure.llm.base import LLMRequestParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedFrameModel:
    """一次解析得到的请求参数与独立输入预算，供同一帧恢复时复用。

    ``params`` 是本帧发给提供商的模型、凭证与采样参数，不含 MTP 停止符；
    输入预算与展示名不属于请求参数，不传给模型 SDK。
    """

    params: LLMRequestParams
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
    根据 agent_profile 从注册表解析 LLM 参数，转换为本帧的请求参数；
    输入预算与展示名独立返回，不传给模型 SDK。

    参数覆盖优先级（从高到低）：
        会话请求 (generation_options) > Agent Profile > 模型注册表定义

    - model：会话传入的注册表 ID 覆盖 profile.model_name（供侧边栏实时切换模型调试）
    - api_key / api_base：来自解析出的模型（Profile 仅通过 model_name 选择模型）
    - temperature / top_p：会话覆盖优先，其次 Profile，最后模型定义默认
    - max_tokens：会话覆盖优先，其次模型定义默认

    设计原则：
    - 注册表未注入（model_registry 为 None）时，由调用方通过 generation_options
      直接传可执行的 model，未提供的采样参数取全局默认值；缺少 model 时抛出
      ``ModelNotFoundError``，与注册表找不到模型同样形成帧失败。
    - 注册表解析失败由 run_frame 稳定投影为 FAILED，不静默降级到其他模型。
    """
    session_opts = dict(generation_options or {})
    if model_registry is None:
        model = session_opts.get("model")
        if not model:
            raise ModelNotFoundError("未注入模型注册表时，生成参数必须提供可执行的 model")
        max_tokens = _or_default(session_opts.get("max_tokens"), DEFAULT_MAX_TOKENS)
        return ResolvedFrameModel(
            params=LLMRequestParams(
                model=model,
                temperature=_or_default(session_opts.get("temperature"), DEFAULT_TEMPERATURE),
                top_p=_or_default(session_opts.get("top_p"), DEFAULT_TOP_P),
                max_tokens=max_tokens,
            ),
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

    # resolve 已完成三级优先级合并，请求参数以其结果为准（会话传入的注册表 ID
    # 已替换为提供商可执行的模型标识符）。api_key / api_base 为 None 是合法
    # 状态，由提供商 SDK 从环境变量读取。
    if not llm_config.model:
        raise ModelNotFoundError(f"模型 '{effective_model_name}' 没有可执行的模型标识符")
    params = LLMRequestParams(
        model=llm_config.model,
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        max_tokens=llm_config.max_tokens,
    )
    logger.debug(
        f"模型解析: effective={effective_model_name!r} "
        f"(session={session_model_id!r}, profile={profile.model_name!r}) → "
        f"litellm_model={llm_config.model!r}, display={display_name!r}, "
        f"temperature={llm_config.temperature}, top_p={llm_config.top_p}"
    )
    return ResolvedFrameModel(
        params=params,
        input_token_limit=_input_token_limit(
            config, llm_config.context_window, llm_config.max_tokens
        ),
        display_name=display_name,
    )


def _or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


def _input_token_limit(
    config: AgentRuntimeConfig, context_window: int | None, max_tokens: int
) -> int:
    """生成预算和安全余量从总窗口扣除；非正预算保留给输入检查明确拒绝。"""
    window = context_window or config.default_context_window
    return window - max_tokens - config.safety_margin
