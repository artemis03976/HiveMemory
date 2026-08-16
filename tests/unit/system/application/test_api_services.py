"""API 应用服务装配测试。"""

from unittest.mock import MagicMock, patch

import pytest

from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.application.memory_service import (
    MemoryApplicationService,
)
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.application.topic_service import TopicApplicationService
from hivememory.system.config.passive import PassiveIngressConfig
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.system import HiveMemorySystem


@pytest.fixture
def passive_config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler
    config.passive_ingress = PassiveIngressConfig()
    return config


class TestApiApplicationServices:
    def test_hivememory_system_build_exposes_api_services(self, passive_config):
        with (
            patch("hivememory.system.assembler.GatewaySystem"),
            patch("hivememory.system.assembler.PatchouliSystem"),
            patch("hivememory.system.assembler.AliceSystem"),
            patch("hivememory.system.assembler.ModelRegistry"),
            patch("hivememory.system.assembler.ProviderRegistry"),
        ):
            system = HiveMemorySystem.build(config=passive_config)

        assert isinstance(system.memory_service, MemoryApplicationService)
        assert isinstance(system.agent_service, AgentApplicationService)
        assert isinstance(system.topic_service, TopicApplicationService)
        assert isinstance(system.readiness_service, SystemReadinessService)
        assert system.memory_service.config is passive_config
        assert system.agent_service.config is passive_config
        assert system.topic_service.config is passive_config

    def test_server_deps_return_api_services(self, passive_config):
        from hivememory.server import deps

        previous_system = deps._system
        try:
            with (
                patch("hivememory.system.assembler.GatewaySystem"),
                patch("hivememory.system.assembler.PatchouliSystem"),
                patch("hivememory.system.assembler.AliceSystem"),
                patch("hivememory.system.assembler.ModelRegistry"),
                patch("hivememory.system.assembler.ProviderRegistry"),
            ):
                system = HiveMemorySystem.build(config=passive_config)
            deps._system = system

            assert deps.get_memory_service() is system.memory_service
            assert deps.get_chat_service() is system.chat_service
            assert deps.get_ingress_service() is system.ingress_service
            assert deps.get_agent_service() is system.agent_service
            assert deps.get_topic_service() is system.topic_service
        finally:
            deps._system = previous_system


