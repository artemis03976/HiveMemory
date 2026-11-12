"""插件服务：命令成功之后才登记接入并交付凭据，来源与 Agent 分开维护。

接入注册表与访问记录是真实的（登记文件在 tmp_path）；harness 的插件命令与 Windows
用户环境变量换成替身：前者由测试决定成败，后者记录写入与删除。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hivememory.config.access import SystemPrincipalsFile, WorkspaceActorsFile, file_digest
from hivememory.core.access import CallerPrincipal, WorkingActor, WorkspaceOperation
from hivememory.core.models import ActorIdentity
from hivememory.core.models.workspace import resolve_default_workspace_identity
from hivememory.system.access import SystemActorAccessRegistry
from hivememory.system.plugins import harnesses
from hivememory.system.plugins.harnesses import CommandResult
from hivememory.system.plugins.service import (
    PLUGIN_OPERATIONS,
    PYTHON_ENV,
    PluginInstallationService,
)
from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
from hivememory.workspace.capability.root.access import RootAccessService
from hivememory.workspace.registry import WorkspaceActorAccessRegistry

_PRINCIPALS_YAML = """\
principals:
  - principal_id: "hivememory:http-server"
    kind: "http-server"
    adapters: ["http"]
    allow_system_actor: true
"""

_ACTORS_YAML = """\
workspace_actors:
  - owner_user_id: "default"
    workspace_id: "main_workspace"
    user_id: "default"
    agent_id: "system"
"""

_PYTHON = r"C:\HiveMemory\resources\python\python.exe"


class _Runner:
    """按命令前缀返回预置结果；其余命令成功，输出为空列表的 JSON。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.failures: dict[tuple[str, ...], CommandResult] = {}

    def fail(self, *prefix: str, stderr: str = "boom") -> None:
        self.failures[prefix] = CommandResult(prefix, 1, "", stderr)

    def __call__(self, argv):
        argv = tuple(argv)
        self.calls.append(argv)
        for prefix, result in self.failures.items():
            if argv[: len(prefix)] == prefix:
                return result
        stdout = '{"marketplaces": []}' if argv[:3] == ("codex", "plugin", "marketplace") else "[]"
        return CommandResult(argv, 0, stdout, "")


class _Environment:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set(self, name: str, value: str) -> None:
        self.values[name] = value

    def delete(self, name: str) -> None:
        self.values.pop(name, None)


def _load(path: Path, text: str, model):
    path.write_text(text, encoding="utf-8")
    return model.model_validate(yaml.safe_load(text)), file_digest(path)


@pytest.fixture
def harness_commands(monkeypatch):
    """claude、codex、pi 都能在 PATH 上找到，命令名就是它们的路径；dsh 桌面端没有装。"""
    monkeypatch.setattr(harnesses.shutil, "which", lambda name: name)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)


@pytest.fixture
def setup(tmp_path, harness_commands):
    principals_file, principals_digest = _load(
        tmp_path / "system_principals.yaml", _PRINCIPALS_YAML, SystemPrincipalsFile
    )
    actors_file, actors_digest = _load(
        tmp_path / "workspace_actors.yaml", _ACTORS_YAML, WorkspaceActorsFile
    )
    principals = SystemActorAccessRegistry.from_config(
        principals_file,
        file_path=tmp_path / "system_principals.yaml",
        file_digest=principals_digest,
        credentials_path=tmp_path / "principals.secrets.yaml",
    )
    actors = WorkspaceActorAccessRegistry.from_config(
        actors_file, file_path=tmp_path / "workspace_actors.yaml", file_digest=actors_digest
    )
    root_access = RootAccessService(
        actors, operation_authorizer=WorkspaceOperationAuthorizer(actors)
    )
    integrations = tmp_path / "integrations"
    (integrations / "hivememory").mkdir(parents=True)
    runner = _Runner()
    environment = _Environment()

    def make(version: str = "0.7.3") -> PluginInstallationService:
        return PluginInstallationService(
            principal_registry=principals,
            root_access=root_access,
            integrations_dir=integrations,
            install_record_path=tmp_path / "plugin_installs.json",
            version=version,
            python_executable=_PYTHON,
            user_environment=environment,
            run=runner,
        )

    actor = WorkingActor(
        principal=CallerPrincipal("hivememory:http-server"),
        actor=ActorIdentity(user_id="default", agent_id="system"),
        workspace=resolve_default_workspace_identity("default"),
    )

    class Setup:
        pass

    s = Setup()
    s.principals, s.root_access, s.runner, s.environment = (
        principals,
        root_access,
        runner,
        environment,
    )
    s.make, s.actor, s.record = make, actor, tmp_path / "plugin_installs.json"
    return s


def _agent_records(setup) -> dict[str | None, frozenset[WorkspaceOperation]]:
    records = setup.root_access.list_records(
        target_workspace=setup.actor.workspace, actor=setup.actor
    )
    return {record.agent_id: frozenset(record.allowed_operations) for record in records}


@pytest.mark.asyncio
async def test_install_registers_the_source_and_delivers_the_credential(setup):
    """命令成功后登记来源（只用生成的凭据），为缺省 Agent 补记录，凭据与解释器写进用户环境变量。"""
    result = await setup.make().install("claude-code", actor=setup.actor)

    assert result.failure is None and result.status.installed
    entry = setup.principals.entry_for("harness:claude-code")
    assert entry.credential_env is None and entry.credential_source == "generated"
    assert entry.allowed_agent_ids == frozenset({"claude_code"})
    assert _agent_records(setup) == {"claude_code": PLUGIN_OPERATIONS}
    assert setup.environment.values == {
        "HIVEMEMORY_CLAUDE_CODE_TOKEN": entry.credential,
        PYTHON_ENV: _PYTHON,
    }
    assert ("claude", "plugin", "install", "hivememory-connector@hivememory-local") in (
        setup.runner.calls
    )


@pytest.mark.asyncio
async def test_failed_or_missing_harness_registers_nothing(setup):
    """插件命令失败或找不到 harness 时不登记来源、不写环境变量，返回失败的命令与输出。"""
    setup.runner.fail("pi", "install", stderr="EACCES: permission denied")

    failed = await setup.make().install("pi", actor=setup.actor)
    missing = await setup.make().install("dsh", actor=setup.actor)

    assert failed.failure.reason == "command_failed"
    assert failed.failure.output_tail == ("EACCES: permission denied",)
    assert missing.failure.reason == "harness_missing"
    assert setup.principals.entry_for("harness:pi") is None
    assert setup.principals.entry_for("harness:dsh") is None
    assert setup.environment.values == {} and _agent_records(setup) == {}


@pytest.mark.asyncio
async def test_reinstall_keeps_the_declarable_agents_credential_and_adjusted_record(setup):
    """重新安装沿用来源：用户加上的 Agent、生成的凭据与调整过的缺省 Agent 记录都不变。"""
    service = setup.make()
    await service.install("codex", actor=setup.actor)
    first = setup.principals.entry_for("harness:codex")
    setup.principals.update(
        "harness:codex",
        adapters=first.adapters,
        allowed_agent_ids=frozenset({"codex", "reviewer"}),
        enabled=True,
    )
    setup.root_access.put_record(
        "codex",
        enabled=True,
        allowed_operations=frozenset({WorkspaceOperation.MEMORY_READ}),
        target_workspace=setup.actor.workspace,
        actor=setup.actor,
    )

    await service.install("codex", actor=setup.actor)

    again = setup.principals.entry_for("harness:codex")
    assert again.allowed_agent_ids == frozenset({"codex", "reviewer"})
    assert again.credential == first.credential
    assert _agent_records(setup) == {"codex": frozenset({WorkspaceOperation.MEMORY_READ})}


@pytest.mark.asyncio
async def test_manual_install_registers_without_running_plugin_commands(setup):
    """用户已经手动装好插件时只登记接入、交付凭据，不运行插件命令。"""
    result = await setup.make().install("pi", actor=setup.actor, run_plugin_commands=False)

    assert result.status.installed and setup.runner.calls == []
    assert "HIVEMEMORY_PI_TOKEN" in setup.environment.values


@pytest.mark.asyncio
async def test_uninstall_removes_the_source_even_when_the_command_fails(setup):
    """卸载命令失败时来源、凭据与环境变量照样删除；Agent 的访问记录保留。"""
    service = setup.make()
    await service.install("claude-code", actor=setup.actor)
    setup.runner.fail("claude", "plugin", "list")

    result = await service.uninstall("claude-code")

    assert result.failure.reason == "command_failed"
    assert setup.principals.entry_for("harness:claude-code") is None
    assert setup.environment.values == {}
    assert _agent_records(setup) == {"claude_code": PLUGIN_OPERATIONS}


@pytest.mark.asyncio
async def test_python_variable_stays_while_another_python_hook_plugin_is_installed(setup):
    """Claude Code 与 Codex 共用 hook 的解释器变量：卸载其中一个时，另一个还在就保留它。"""
    service = setup.make()
    await service.install("claude-code", actor=setup.actor)
    await service.install("codex", actor=setup.actor)

    await service.uninstall("codex")

    assert setup.environment.values[PYTHON_ENV] == _PYTHON


@pytest.mark.asyncio
async def test_upgrade_reinstalls_copied_plugins_recorded_at_an_older_version(setup):
    """升级后只重新安装记录版本落后的复制安装插件；链接安装的插件不动。"""
    await setup.make("0.7.3").install("claude-code", actor=setup.actor)
    await setup.make("0.7.3").install("pi", actor=setup.actor)
    upgraded = setup.make("0.7.4")
    assert {s.harness.harness_id for s in upgraded.statuses() if s.needs_update} == {"claude-code"}
    setup.runner.calls.clear()

    await upgraded.update_copied_plugins()

    assert {call[0] for call in setup.runner.calls} == {"claude"}
    assert json.loads(setup.record.read_text(encoding="utf-8"))["claude-code"] == "0.7.4"
    assert not any(s.needs_update for s in upgraded.statuses())


@pytest.mark.asyncio
async def test_install_without_user_environment_is_unavailable(setup):
    """不在 Windows 上（没有用户环境变量可写）时不能代为安装，什么都不登记。"""
    service = PluginInstallationService(
        principal_registry=setup.principals,
        root_access=setup.root_access,
        integrations_dir=Path("unused"),
        install_record_path=setup.record,
        version="0.7.3",
        python_executable=_PYTHON,
        user_environment=None,
        run=setup.runner,
    )

    result = await service.install("claude-code", actor=setup.actor)

    assert result.failure.reason == "unavailable"
    assert setup.principals.entry_for("harness:claude-code") is None
    assert setup.runner.calls == []
