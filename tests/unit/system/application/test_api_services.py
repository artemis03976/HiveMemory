"""API 应用服务装配测试。"""

from unittest.mock import MagicMock, patch

import pytest

from hivememory.config.passive import PassiveIngressConfig
from hivememory.config.workspace import WorkspaceConfig
from hivememory.system.application.readiness_service import SystemReadinessService
from hivememory.system.system import HiveMemorySystem
from hivememory.workspace.authentication import ActorAuthenticationGateway
from hivememory.workspace.capability.root.agent_profiles import RootAgentProfileService
from hivememory.workspace.capability.root.assets import RootWorkspaceAssetService
from hivememory.workspace.capability.root.memory import RootMemoryService
from hivememory.workspace.capability.root.memory_tasks import RootMemoryTaskService
from hivememory.workspace.capability.root.topic import RootTopicService


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
    config.workspace = WorkspaceConfig()
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

        assert isinstance(system.root_memory_service, RootMemoryService)
        assert isinstance(system.root_agent_service, RootAgentProfileService)
        assert isinstance(system.root_topic_service, RootTopicService)
        assert isinstance(system.root_memory_task_service, RootMemoryTaskService)
        assert isinstance(system.root_workspace_asset_service, RootWorkspaceAssetService)
        assert isinstance(system.readiness_service, SystemReadinessService)
        # A1 访问边界返工：统一认证网关随组合根装配并暴露给 server 入口。
        assert isinstance(system.access_gateway, ActorAuthenticationGateway)

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

            assert deps.get_root_memory_service() is system.root_memory_service
            assert deps.get_process_service() is system.process_service
            assert deps.get_ingress_service() is system.ingress_service
            assert deps.get_root_agent_service() is system.root_agent_service
            assert deps.get_root_topic_service() is system.root_topic_service
            assert deps.get_root_memory_task_service() is system.root_memory_task_service
            assert deps.get_root_workspace_asset_service() is system.root_workspace_asset_service
        finally:
            deps._system = previous_system
