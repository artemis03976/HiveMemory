import pytest

from hivememory.core.models import Identity
from hivememory.gateway.commands import (
    CommandCategory,
    CommandDefinition,
    CommandExecutionResult,
    CommandExecutionStatus,
    CommandParseResult,
    CommandParseStatus,
    CommandPermissionPolicy,
    CommandRegistry,
    CommandRouteTarget,
    CommandRouteTargetKind,
    SystemCommandDispatcher,
    create_builtin_command_registry,
)


def _definition(
    command_id: str = "system.test",
    primary_name: str = "/test",
    **kwargs,
) -> CommandDefinition:
    defaults = dict(
        category=CommandCategory.SYSTEM,
        summary="Test command",
        route_target=CommandRouteTarget(name=command_id),
    )
    defaults.update(kwargs)
    return CommandDefinition(
        command_id=command_id,
        primary_name=primary_name,
        **defaults,
    )


def _parse_result(
    command_id: str = "system.test",
    name: str = "/test",
    status: CommandParseStatus = CommandParseStatus.MATCHED,
    **kwargs,
) -> CommandParseResult:
    defaults = dict(
        raw_input=name,
        name=name,
        tokens=[name],
        matched_alias=name,
        parse_status=status,
    )
    defaults.update(kwargs)
    return CommandParseResult(command_id=command_id, **defaults)


@pytest.mark.asyncio
async def test_help_returns_visible_commands():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/help"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert "/help" in result.message
    assert "/commands" in result.message
    assert result.data["commands"]


@pytest.mark.asyncio
async def test_commands_returns_structured_listing():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/commands"))

    assert result.status == CommandExecutionStatus.COMPLETED
    command_ids = {item["command_id"] for item in result.data["commands"]}
    assert {"system.help", "system.commands", "system.clear", "runtime.status"} <= command_ids


@pytest.mark.asyncio
async def test_commands_listing_can_be_hidden():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry, expose_listing=False)

    result = await dispatcher.execute(registry.match("/commands"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert result.data["commands"] == []


@pytest.mark.asyncio
async def test_clear_returns_client_action_without_backend_side_effect():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/clear"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert result.client_action == {"type": "clear_chat"}


@pytest.mark.asyncio
async def test_status_returns_gateway_summary():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/status"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert result.data["gateway"] == "ok"
    assert result.data["debug_enabled"] is False


@pytest.mark.asyncio
async def test_unknown_parse_result_is_rejected():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)
    command = CommandParseResult(
        raw_input="/missing",
        name="/missing",
        tokens=["/missing"],
        parse_status=CommandParseStatus.UNKNOWN,
        error="Unknown system command: /missing",
    )

    result = await dispatcher.execute(command)

    assert result.status == CommandExecutionStatus.REJECTED
    assert result.error_code == "command.parse.unknown"


@pytest.mark.asyncio
async def test_invalid_parse_result_is_rejected():
    registry = create_builtin_command_registry()
    dispatcher = SystemCommandDispatcher(registry)
    command = CommandParseResult(
        raw_input='/help "',
        name="/help",
        tokens=[],
        parse_status=CommandParseStatus.INVALID_ARGS,
        error="unterminated quote",
    )

    result = await dispatcher.execute(command)

    assert result.status == CommandExecutionStatus.REJECTED
    assert result.error_code == "command.parse.invalid_args"


@pytest.mark.asyncio
async def test_unregistered_command_id_is_rejected():
    dispatcher = SystemCommandDispatcher(CommandRegistry())

    result = await dispatcher.execute(_parse_result(command_id="system.missing"))

    assert result.status == CommandExecutionStatus.REJECTED
    assert result.error_code == "command.not_registered"


@pytest.mark.asyncio
async def test_debug_command_requires_debug_enabled():
    registry = CommandRegistry()
    registry.register(
        _definition(
            command_id="debug.inspect",
            primary_name="/debug",
            permission=CommandPermissionPolicy(visibility="debug"),
        )
    )
    hidden_dispatcher = SystemCommandDispatcher(registry)
    enabled_dispatcher = SystemCommandDispatcher(registry, debug_enabled=True)

    rejected = await hidden_dispatcher.execute(registry.match("/debug"))
    completed = await enabled_dispatcher.execute(registry.match("/debug"))

    assert rejected.status == CommandExecutionStatus.REJECTED
    assert completed.status == CommandExecutionStatus.NOT_IMPLEMENTED


@pytest.mark.asyncio
async def test_admin_command_requires_allowlist():
    registry = CommandRegistry()
    registry.register(
        _definition(
            command_id="admin.inspect",
            primary_name="/admin",
            permission=CommandPermissionPolicy(
                visibility="admin",
                allowed_user_ids=["trusted-user"],
            ),
        )
    )
    dispatcher = SystemCommandDispatcher(registry)

    rejected = await dispatcher.execute(registry.match("/admin"), identity=Identity(user_id="other"))
    permitted = await dispatcher.execute(registry.match("/admin"), identity=Identity(user_id="trusted-user"))

    assert rejected.status == CommandExecutionStatus.REJECTED
    assert permitted.status == CommandExecutionStatus.NOT_IMPLEMENTED


@pytest.mark.asyncio
async def test_destructive_command_requires_confirmation():
    registry = CommandRegistry()
    registry.register(
        _definition(
            command_id="system.destroy",
            primary_name="/destroy",
            permission=CommandPermissionPolicy(destructive=True),
        )
    )
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/destroy"))

    assert result.status == CommandExecutionStatus.REQUIRES_CONFIRMATION
    assert result.error_code == "command.requires_confirmation"


@pytest.mark.asyncio
async def test_missing_local_handler_returns_not_implemented():
    registry = CommandRegistry()
    registry.register(_definition())
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/test"))

    assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.error_code == "command.handler.not_implemented"


@pytest.mark.asyncio
async def test_global_route_without_bus_returns_not_implemented():
    registry = CommandRegistry()
    registry.register(
        _definition(
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                name="system.route",
            )
        )
    )
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/test"))

    assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.error_code == "command.global_route.no_bus"


@pytest.mark.asyncio
async def test_global_route_with_bus_wraps_response():
    class FakeBus:
        def __init__(self):
            self.calls = []

        async def request(self, route, **kwargs):
            self.calls.append((route, kwargs))
            return {"ok": True}

    registry = CommandRegistry()
    registry.register(
        _definition(
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                name="system.route",
                payload_template={"source": "test"},
            )
        )
    )
    bus = FakeBus()
    dispatcher = SystemCommandDispatcher(registry, global_bus=bus)

    result = await dispatcher.execute(registry.match("/test --limit=1"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert result.data == {"response": {"ok": True}}
    assert bus.calls[0][0] == "system.route"
    assert bus.calls[0][1]["args"] == {"limit": "1"}
    assert bus.calls[0][1]["source"] == "test"


@pytest.mark.asyncio
async def test_global_route_can_return_execution_result():
    class FakeBus:
        async def request(self, route, **kwargs):
            return CommandExecutionResult(
                command_id="system.test",
                status=CommandExecutionStatus.COMPLETED,
                message="done",
            )

    registry = CommandRegistry()
    registry.register(
        _definition(
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.GLOBAL_ROUTE,
                name="system.route",
            )
        )
    )
    dispatcher = SystemCommandDispatcher(registry, global_bus=FakeBus())

    result = await dispatcher.execute(registry.match("/test"))

    assert result.status == CommandExecutionStatus.COMPLETED
    assert result.message == "done"


@pytest.mark.asyncio
async def test_future_job_route_returns_not_implemented():
    registry = CommandRegistry()
    registry.register(
        _definition(
            route_target=CommandRouteTarget(
                kind=CommandRouteTargetKind.FUTURE_JOB,
                name="future.job",
            )
        )
    )
    dispatcher = SystemCommandDispatcher(registry)

    result = await dispatcher.execute(registry.match("/test"))

    assert result.status == CommandExecutionStatus.NOT_IMPLEMENTED
    assert result.error_code == "command.route.not_implemented"


@pytest.mark.asyncio
async def test_handler_exception_returns_failed():
    registry = CommandRegistry()
    registry.register(_definition(route_target=CommandRouteTarget(name="system.fail")))

    def failing_handler(**kwargs):
        raise RuntimeError("boom")

    dispatcher = SystemCommandDispatcher(
        registry,
        handlers={"system.fail": failing_handler},
    )

    result = await dispatcher.execute(registry.match("/test"))

    assert result.status == CommandExecutionStatus.FAILED
    assert result.error_code == "command.failed"
