"""Agent 执行侧的模型解析端口，以及帧的请求参数与输入预算。

执行侧向注册表提两个问题："这个 ID 指的是哪个模型"（取模型定义，读上下文窗口
与展示名），以及"按这个选择调用模型要发什么参数"（取请求参数）。实现方是 System
的模型注册表（``system.model_registry.ModelRegistry``），由组合根注入。找不到模型
时实现方抛出 ``core.errors.ModelNotFoundError``。
"""

from __future__ import annotations

import logging
from typing import Protocol

from hivememory.config.alice import AgentRuntimeConfig
from hivememory.core.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from hivememory.core.errors import ModelNotFoundError
from hivememory.core.models import AgentProfile
from hivememory.core.models.model_definition import ModelDefinition, ModelSelection
from hivememory.infrastructure.llm.base import LLMRequestParams

logger = logging.getLogger(__name__)


class ModelResolver(Protocol):
    """模型注册表的端口：按 ID 取模型定义，按选择取请求参数。"""

    def select_model(self, model_id: str) -> ModelDefinition:
        """按 ID 选出模型定义；``default`` 选出注册表的默认模型。"""
        ...

    def resolve(self, selection: ModelSelection) -> LLMRequestParams:
        """把选择解析为请求参数：覆盖值优先，为 None 时取模型定义的默认值。"""
        ...


def resolve_frame_model(
    profile: AgentProfile,
    model_selection: ModelSelection | None,
    *,
    model_registry: ModelResolver | None,
    config: AgentRuntimeConfig,
) -> tuple[LLMRequestParams, int, str]:
    """
    解析本帧的模型，返回 ``(请求参数, 输入上限, 展示名)``。

    请求参数随帧保留、发给提供商（不含 MTP 停止符）；输入上限与展示名不属于
    请求参数，不传给模型 SDK。

    ``model_selection`` 是会话请求这一层的选择，未给出的项沿用 Agent Profile，
    Profile 也未给出的生成参数由注册表回落到模型定义：

    - model_id：会话传入的注册表 ID 覆盖 profile.model_name（供侧边栏实时切换模型调试）
    - api_key / api_base：来自解析出的模型（Profile 仅通过 model_name 选择模型）
    - temperature / top_p：会话覆盖优先，其次 Profile，最后模型定义默认
    - max_tokens：会话覆盖优先，其次模型定义默认

    设计原则：
    - 注册表未注入（model_registry 为 None）时，会话选择的 model_id 直接作为可执行的
      模型标识符，未提供的采样参数取全局默认值；缺少 model_id 时抛出
      ``ModelNotFoundError``，与注册表找不到模型同样形成帧失败。
    - 注册表解析失败由 run_frame 稳定投影为 FAILED，不静默降级到其他模型。
    """
    session = model_selection or ModelSelection()
    if model_registry is None:
        if not session.model_id:
            raise ModelNotFoundError("未注入模型注册表时，会话选择必须提供可执行的 model_id")
        max_tokens = _or_default(session.max_tokens, DEFAULT_MAX_TOKENS)
        params = LLMRequestParams(
            model=session.model_id,
            temperature=_or_default(session.temperature, DEFAULT_TEMPERATURE),
            top_p=_or_default(session.top_p, DEFAULT_TOP_P),
            max_tokens=max_tokens,
        )
        return params, _input_token_limit(config, None, max_tokens), ""

    # 会话传入的注册表 ID 与 Profile 的 model_name 语义一致（都是注册表 ID / 'default'）。
    temperature = session.temperature
    if temperature is None:
        temperature = profile.temperature
    top_p = session.top_p
    if top_p is None:
        top_p = profile.top_p
    selection = ModelSelection(
        model_id=session.model_id or profile.model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=session.max_tokens,
    )
    # 两次查询之间没有 await，'default' 不会在中途指向另一个模型。
    model_definition = model_registry.select_model(selection.model_id or "default")
    params = model_registry.resolve(selection)
    logger.debug(
        "模型解析: selection=%r (profile=%r) → litellm_model=%r, display=%r, temperature=%s, top_p=%s",
        selection.model_id,
        profile.model_name,
        params.model,
        model_definition.display_name,
        params.temperature,
        params.top_p,
    )
    assert params.max_tokens is not None  # 注册表以覆盖值或模型默认值填入
    limit = _input_token_limit(config, model_definition.context_window, params.max_tokens)
    return params, limit, model_definition.display_name


def _or_default[T](value: T | None, default: T) -> T:
    return default if value is None else value


def _input_token_limit(
    config: AgentRuntimeConfig, context_window: int | None, max_tokens: int
) -> int:
    """生成预算和安全余量从总窗口扣除；非正预算保留给输入检查明确拒绝。"""
    window = context_window or config.default_context_window
    return window - max_tokens - config.safety_margin
