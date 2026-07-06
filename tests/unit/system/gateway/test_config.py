from hivememory.system.config import (
    HiveMemoryConfig,
    PatchouliConfig,
    RuleInterceptorConfig,
    SystemCommandConfig,
    SystemGatewayConfig,
)


def test_system_gateway_config_defaults():
    config = SystemGatewayConfig()

    assert isinstance(config.interceptor, RuleInterceptorConfig)
    assert config.interceptor.enabled is True
    assert config.analyzer.enabled is True
    assert isinstance(config.commands, SystemCommandConfig)
    assert config.commands.enabled is True
    assert config.commands.unknown_command_policy == "reject"
    assert config.commands.expose_listing is True
    assert config.commands.enable_debug_commands is False
    assert config.commands.builtin == {}


def test_hivememory_config_accepts_top_level_gateway():
    config = HiveMemoryConfig.model_validate(
        {
            "gateway": {
                "interceptor": {"enabled": False},
                "analyzer": {"enabled": False},
                "commands": {
                    "enabled": False,
                    "unknown_command_policy": "ignore",
                    "expose_listing": False,
                    "enable_debug_commands": True,
                    "builtin": {"system.clear": False},
                },
            },
        }
    )

    assert config.gateway.interceptor.enabled is False
    assert config.gateway.analyzer.enabled is False
    assert config.gateway.commands.enabled is False
    assert config.gateway.commands.unknown_command_policy == "ignore"
    assert config.gateway.commands.expose_listing is False
    assert config.gateway.commands.enable_debug_commands is True
    assert config.gateway.commands.builtin == {"system.clear": False}


def test_patchouli_config_no_longer_owns_gateway_config():
    config = PatchouliConfig()

    assert not hasattr(config, "gateway")
