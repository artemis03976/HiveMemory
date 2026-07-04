from hivememory.system.config import (
    HiveMemoryConfig,
    PatchouliConfig,
    RuleInterceptorConfig,
    SystemGatewayConfig,
)


def test_system_gateway_config_defaults():
    config = SystemGatewayConfig()

    assert isinstance(config.interceptor, RuleInterceptorConfig)
    assert config.interceptor.enabled is True
    assert config.analyzer.enabled is True


def test_hivememory_config_accepts_top_level_gateway():
    config = HiveMemoryConfig.model_validate(
        {
            "gateway": {
                "interceptor": {"enabled": False},
                "analyzer": {"enabled": False},
            },
        }
    )

    assert config.gateway.interceptor.enabled is False
    assert config.gateway.analyzer.enabled is False


def test_patchouli_config_no_longer_owns_gateway_config():
    config = PatchouliConfig()

    assert not hasattr(config, "gateway")
