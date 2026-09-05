"""系统指令 dispatcher 执行路径测试。"""

import pytest

from hivememory.core.models import ActorIdentity
from hivememory.core.protocol.gateway import (
    CommandExecutionResult,
    CommandExecutionStatus,
)
from hivememory.gateway.commands.builtins import create_builtin_command_registry
from hivememory.gateway.commands.dispatcher import SystemCommandDispatcher
from hivememory.gateway.commands.models import (
    CommandCategory,
    CommandDefinition,
    CommandParseResult,
    CommandParseStatus,
    CommandPermissionPolicy,
    CommandRouteTarget,
    CommandRouteTargetKind,
)
from hivememory.gateway.commands.registry import CommandRegistry


def _definition(
    *,
    command_id: str,
    primary_name: str,
    kind: CommandRouteTargetKind = CommandRouteTargetKind.LOCAL_HANDLER,
    target_name: str | None = None,
    visibility: str = "public",
    allowed_user_ids: tuple[str, ...] | None = None,
    allowed_agent_ids: tuple[str, ...] | None = None,
    requires_confirmation: bool = False,
    destructive: bool = False,
    payload_template: dict | None = None,
) -> CommandDefinition:
    return CommandDefinition(
        command_id=command_id,
        category=CommandCategory.SYSTEM,
        primary_name=primary_name,
        summary="summary",
        permission=CommandPermissionPolicy(
            visibility=visibility,
            allowed_user_ids=allowed_user_ids,
            allowed_agent_ids=allowed_agent_ids,
            requires_confirmation=requires_confirmation,
            destructive=destructive,
        ),
        route_target=CommandRouteTarget(
            kind=kind,
            name=target_name or command_id,
            payload_template=payload_template or {},
        ),
    )


def _parse(command_id: str | None = None, status: CommandParseStatus = CommandParseStatus.MATCHED):
    return CommandParseResult(
        command_id=command_id,
        raw_input="/cmd",
        parse_status=status,
    )


@pytest.mark.asyncio
class TestDispatcherRejections:
    async def test_none_command_rejected(self):
        dispatcher = SystemCommandDispatcher(CommandRegistry())
        result = await dispatcher.execute(None)
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.missing"

    async def test_not_matched_rejected(self):
        dispatcher = SystemCommandDispatcher(CommandRegistry())
        result = await dispatcher.execute(
            _parse(status=CommandParseStatus.UNKNOWN)
        )
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.parse.unknown"

    async def test_missing_command_id_rejected(self):
        dispatcher = SystemCommandDispatcher(CommandRegistry())
        result = await dispatcher.execute(_parse(command_id=None))
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.missing_id"

    async def test_unregistered_command_rejected(self):
        dispatcher = SystemCommandDispatcher(CommandRegistry())
        result = await dispatcher.execute(_parse(command_id="nope.cmd"))
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.not_registered"


@pytest.mark.asyncio
class TestDispatcherPermissions:
    async def test_user_not_allowed(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="adm.cmd",
                primary_name="/adm",
                visibility="admin",
                allowed_user_ids=("admin_u",),
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(
            _parse(command_id="adm.cmd"),
            identity=ActorIdentity(user_id="other"),
        )
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.permission_denied"

    async def test_agent_not_allowed(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="agent.cmd",
                primary_name="/agent",
                allowed_agent_ids=("allowed_agent",),
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(
            _parse(command_id="agent.cmd"),
            identity=ActorIdentity(agent_id="other_agent"),
        )
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.permission_denied"

    async def test_debug_disabled(self):
        registry = CommandRegistry()
        registry.register(
            _definition(command_id="dbg.cmd", primary_name="/dbg", visibility="debug")
        )
        dispatcher = SystemCommandDispatcher(registry, debug_enabled=False)
        result = await dispatcher.execute(_parse(command_id="dbg.cmd"))
        assert result.status == CommandExecutionStatus.REJECTED
        assert result.error_code == "command.permission_denied"

    async def test_debug_enabled_allows(self):
        registry = CommandRegistry()
        registry.register(
            _definition(command_id="dbg.cmd", primary_name="/dbg", visibility="debug")
        )

        def dbg_handler(**kwargs):
            return CommandExecutionResult(
                command_id="dbg.cmd",
                status=CommandExecutionStatus.COMPLETED,
                message="debug ok",
            )

        dispatcher = SystemCommandDispatcher(
            registry,
            debug_enabled=True,
            handlers={"dbg.cmd": dbg_handler},
        )
        result = await dispatcher.execute(_parse(command_id="dbg.cmd"))
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.message == "debug ok"

    async def test_requires_confirmation(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="danger.cmd",
                primary_name="/danger",
                requires_confirmation=True,
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="danger.cmd"))
        assert result.status == CommandExecutionStatus.REQUIRES_CONFIRMATION

    async def test_destructive_requires_confirmation(self):
        registry = CommandRegistry()
        registry.register(
            _definition(command_id="del.cmd", primary_name="/del", destructive=True)
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="del.cmd"))
        assert result.status == CommandExecutionStatus.REQUIRES_CONFIRMATION


@pytest.mark.asyncio
class TestDispatcherDispatch:
    async def test_local_handler_success(self):
        registry = create_builtin_command_registry()
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="system.help"))
        assert result.status == CommandExecutionStatus.COMPLETED
        assert "/help" in result.message

    async def test_local_handler_missing(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="orphan.cmd",
                primary_name="/orphan",
                target_name="no.such.handler",
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="orphan.cmd"))
        assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
        assert result.error_code == "command.handler.not_implemented"

    async def test_client_action(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="clear.cmd",
                primary_name="/clear",
                kind=CommandRouteTargetKind.CLIENT_ACTION,
                target_name="clear_chat",
                payload_template={"keep_system": True},
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="clear.cmd"))
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.client_action == {"type": "clear_chat", "keep_system": True}

    async def test_global_route_without_bus(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="route.cmd",
                primary_name="/route",
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                target_name="some.route",
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="route.cmd"))
        assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
        assert result.error_code == "command.global_route.no_bus"

    async def test_global_route_with_bus(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="route.cmd",
                primary_name="/route",
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                target_name="some.route",
                payload_template={"extra": "x"},
            )
        )

        class FakeBus:
            def __init__(self):
                self.requests = []

            async def request(self, name, **kwargs):
                self.requests.append((name, kwargs))
                return {"ok": True}

        bus = FakeBus()
        dispatcher = SystemCommandDispatcher(registry, global_bus=bus)
        result = await dispatcher.execute(
            _parse(command_id="route.cmd"),
            identity=ActorIdentity(user_id="u1"),
        )
        assert result.status == CommandExecutionStatus.COMPLETED
        assert result.data["response"] == {"ok": True}
        route_name, kwargs = bus.requests[0]
        assert route_name == "some.route"
        assert kwargs["identity"].user_id == "u1"
        assert kwargs["extra"] == "x"

    async def test_global_route_returns_execution_result(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="route.cmd",
                primary_name="/route",
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                target_name="some.route",
            )
        )

        class FakeBus:
            async def request(self, name, **kwargs):
                return CommandExecutionResult(
                    command_id="route.cmd",
                    status=CommandExecutionStatus.FAILED,
                    message="upstream failed",
                )

        dispatcher = SystemCommandDispatcher(registry, global_bus=FakeBus())
        result = await dispatcher.execute(_parse(command_id="route.cmd"))
        assert result.status == CommandExecutionStatus.FAILED
        assert result.message == "upstream failed"

    async def test_future_job_not_implemented(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="job.cmd",
                primary_name="/job",
                kind=CommandRouteTargetKind.FUTURE_JOB,
            )
        )
        dispatcher = SystemCommandDispatcher(registry)
        result = await dispatcher.execute(_parse(command_id="job.cmd"))
        assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
        assert result.error_code == "command.route.not_implemented"

    async def test_handler_exception_fails(self):
        registry = CommandRegistry()
        registry.register(
            _definition(
                command_id="boom.cmd",
                primary_name="/boom",
                target_name="boom.handler",
            )
        )

        def boom(**kwargs):
            raise RuntimeError("handler exploded")

        dispatcher = SystemCommandDispatcher(
            registry,
            handlers={"boom.handler": boom},
        )
        result = await dispatcher.execute(_parse(command_id="boom.cmd"))
        assert result.status == CommandExecutionStatus.FAILED
        assert result.error_code == "command.failed"
        assert "handler exploded" in result.message

    async def test_custom_handler_override(self):
        registry = create_builtin_command_registry()
        seen = {}

        def custom_help(**kwargs):
            seen["called"] = True
            return CommandExecutionResult(
                command_id="system.help",
                status=CommandExecutionStatus.COMPLETED,
                message="custom help",
            )

        dispatcher = SystemCommandDispatcher(registry, handlers={"system.help": custom_help})
        result = await dispatcher.execute(_parse(command_id="system.help"))
        assert result.message == "custom help"
        assert seen["called"] is True
