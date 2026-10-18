"""组合根的 LLM 装配：共享服务与各组件独立的请求参数。"""

from hivememory.config.app import HiveMemoryConfig
from hivememory.config.shared import LLMConfig
from hivememory.infrastructure.llm.base import LLMRequestParams
from hivememory.system.assembler import SystemAssembler, component_llm_params


def test_gateway_and_librarian_get_request_params_from_their_own_config():
    """两个组件的模型、凭证与采样参数只来自各自的配置段，组件侧不发送 top_p。"""
    config = HiveMemoryConfig(runtime_events={"enabled": False})
    config.shared.llm.gateway = LLMConfig(
        model="gateway-model",
        api_key="gateway-key",
        api_base="https://gateway.invalid",
        temperature=0.0,
        max_tokens=512,
        top_p=0.5,
    )
    config.shared.llm.librarian = LLMConfig(
        model="librarian-model", api_key="librarian-key", temperature=0.3, max_tokens=8192
    )
    assembler = SystemAssembler(config)
    assembler._build_registries()

    llm = assembler._build_llm()

    assert llm.gateway_params == LLMRequestParams(
        model="gateway-model",
        api_key="gateway-key",
        api_base="https://gateway.invalid",
        temperature=0.0,
        max_tokens=512,
    )
    assert llm.librarian_params == LLMRequestParams(
        model="librarian-model", api_key="librarian-key", temperature=0.3, max_tokens=8192
    )


def test_component_without_model_gets_no_request_params():
    """未配置模型的组件不取得请求参数，由组件按无 LLM 降级而不是装配失败。"""
    assert component_llm_params(LLMConfig(model=None)) is None
