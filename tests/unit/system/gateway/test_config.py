from hivememory.system.config import (
    HiveMemoryConfig,
    MemoryGatewayConfig,
    RuleInterceptorConfig,
    SystemGatewayConfig,
    _migrate_legacy_gateway_data,
)


def test_system_gateway_config_defaults():
    config = SystemGatewayConfig()

    assert isinstance(config.interceptor, RuleInterceptorConfig)
    assert config.interceptor.enabled is True
    assert config.analyzer.enabled is True


def test_memory_gateway_config_is_compatible_alias():
    config = MemoryGatewayConfig()

    assert isinstance(config, SystemGatewayConfig)


def test_legacy_patchouli_gateway_migrates_to_top_level_gateway():
    migrated = _migrate_legacy_gateway_data(
        {
            "patchouli": {
                "gateway": {
                    "interceptor": {"enabled": False},
                    "analyzer": {"enabled": False},
                }
            }
        }
    )
    config = HiveMemoryConfig.model_validate(migrated)

    assert config.gateway.interceptor.enabled is False
    assert config.gateway.analyzer.enabled is False


def test_top_level_gateway_wins_over_legacy_patchouli_gateway():
    config = HiveMemoryConfig.model_validate(
        {
            "gateway": {
                "interceptor": {"enabled": True},
                "analyzer": {"enabled": True},
            },
            "patchouli": {
                "gateway": {
                    "interceptor": {"enabled": False},
                    "analyzer": {"enabled": False},
                }
            },
        }
    )

    assert config.gateway.interceptor.enabled is True
    assert config.gateway.analyzer.enabled is True
